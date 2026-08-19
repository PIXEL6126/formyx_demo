"""
formyx_backend/main.py
-----------------------
Top-level entry point for the Formyx Autonomous Drone backend.

Orchestrates all subsystems:
  • MAVLink telemetry (thread-safe connection to ArduPilot)
  • RealSense D435i depth + RGB acquisition
  • Dual-class ONNX perception (balloon + drone)
  • 3D Kalman target tracker
  • OpenCV display window (optional, requires DISPLAY)

Usage
-----
    cd formyx_backend
    python main.py [--connection <conn_str>] [--log-level DEBUG|INFO|WARNING]
                   [--no-display] [--inference-interval N]

SITL example:
    python main.py --connection udpin:localhost:14550

Hardware (Raspberry Pi 5):
    python main.py --connection serial:/dev/ttyAMA0:921600

Camera only (no FC):
    python main.py --no-mavlink
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

from config import load_config, get
from depth.realsense_interface import RealSenseInterface
from perception.detector import ObjectDetector
from tracking.multi_target_tracker import MultiTargetTracker


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def _make_writer(output_dir: str, width: int, height: int) -> tuple[cv2.VideoWriter | None, str | None]:
    from datetime import datetime
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"session_{stamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 15.0, (width, height))
    if not writer.isOpened():
        logging.getLogger(__name__).warning("VideoWriter failed to open — recording disabled.")
        return None, None
    logging.getLogger(__name__).info("Recording video to: %s", out_path)
    return writer, out_path


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_signal(signum, frame):  # noqa: ANN001
    global _shutdown_requested
    logging.getLogger(__name__).warning(
        "Signal %d received — initiating clean shutdown.", signum
    )
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="formyx_backend",
        description="Formyx Autonomous Balloon-Tracking Drone Backend",
    )
    parser.add_argument(
        "--connection",
        default=None,
        help="MAVLink connection string (overrides settings.yaml). "
             "E.g. udpin:localhost:14550 or serial:/dev/ttyAMA0:921600",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity (default: INFO)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV camera window (useful when running headless)",
    )
    parser.add_argument(
        "--no-mavlink",
        action="store_true",
        help="Skip MAVLink connection (camera + perception only)",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=2,
        help="Run YOLO detection every N frames (default: 2). "
             "1 = every frame, 2 = every other frame for higher FPS.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable saving video recording",
    )
    parser.add_argument(
        "--no-tiled",
        action="store_true",
        help="Disable tiled/SAHI inference (faster, less long-range accuracy)",
    )
    parser.add_argument(
        "--output-dir",
        default="camrec",
        help="Directory to save recorded videos (default: camrec)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MAVLink telemetry thread (non-blocking)
# ---------------------------------------------------------------------------

def _mavlink_thread(conn_string: str, stop_event: threading.Event) -> None:
    log = logging.getLogger("mavlink_thread")
    try:
        from mavlink_interface.connection import MAVLinkConnection
        conn = MAVLinkConnection(connection_string=conn_string)
        conn.connect()
        log.info("MAVLink connected.")
        while not stop_event.is_set():
            telem = conn.get_telemetry()
            log.info(
                "| Armed=%-5s | Mode=%-15s | Alt=%.1fm | Bat=%d%% | "
                "GPS_Fix=%d Sats=%d | GndSpd=%.1fm/s |",
                telem.armed,
                telem.flight_mode,
                telem.alt_agl_m,
                telem.battery_remaining_pct,
                telem.gps_fix_type,
                telem.satellites_visible,
                telem.groundspeed_ms,
            )
            stop_event.wait(timeout=0.1)
        conn.close()
    except Exception as exc:
        log.error("MAVLink thread error: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg = load_config()
    log.info("Formyx Backend starting — loading configuration.")

    # ------------------------------------------------------------------
    # 1. MAVLink connection (direct synchronous creation, async internal loop)
    # ------------------------------------------------------------------
    conn = None
    if not args.no_mavlink:
        conn_string = args.connection or cfg["mavlink"]["connection_string"]
        log.info("Connecting to autopilot at: %s", conn_string)
        from mavlink_interface.connection import MAVLinkConnection
        conn = MAVLinkConnection(connection_string=conn_string)
        try:
            conn.connect()
            log.info("MAVLink connected successfully.")
        except Exception as exc:
            log.error("Failed to connect to MAVLink: %s. Continuing in MOCK/disconnected telemetry mode.", exc)
            conn = None

    # ------------------------------------------------------------------
    # 2. RealSense camera
    # ------------------------------------------------------------------
    log.info("Initialising RealSense D435i...")
    camera = RealSenseInterface(use_mock=False)
    camera.start()
    log.info(
        "Camera ready — intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
        camera.fx, camera.fy, camera.ppx, camera.ppy,
    )

    # ------------------------------------------------------------------
    # 3. Perception — dual-class ONNX detector
    # ------------------------------------------------------------------
    log.info("Loading dual-class YOLO detector (balloon + drone)...")
    enable_tiled = not getattr(args, 'no_tiled', False)
    detector = ObjectDetector(enable_tiled=enable_tiled)
    log.info("Detector ready (ONNX=%s, tiled=%s).", detector.use_onnx, enable_tiled)

    # ------------------------------------------------------------------
    # 4. Multi-target 3D Kalman tracker
    # ------------------------------------------------------------------
    tracker = MultiTargetTracker()

    # ------------------------------------------------------------------
    # 5. BlackBox logger
    # ------------------------------------------------------------------
    from logging_system.logger import BlackBoxLogger
    logger = BlackBoxLogger()
    logger.start()
    last_log_time = 0.0

    # ------------------------------------------------------------------
    # 6. Video Recorder
    # ------------------------------------------------------------------
    video_writer = None
    rec_path = None
    if not args.no_record:
        video_writer, rec_path = _make_writer(args.output_dir, 640, 480)

    # ------------------------------------------------------------------
    # 7. Display window
    # ------------------------------------------------------------------
    show_display = not args.no_display
    window_name = "Formyx Perception — Press Q to quit"
    if show_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------
    # 8. Main perception + tracking loop
    # ------------------------------------------------------------------
    log.info("Entering perception loop. Press Ctrl-C or Q in window to stop.")
    inference_interval = args.inference_interval
    loop_idx = 0
    fps_display = 0.0
    fps_frame_count = 0
    fps_timer = time.monotonic()
    last_detections: list = []

    try:
        while not _shutdown_requested:
            t_loop_start = time.monotonic()
            loop_idx += 1
            run_yolo = (loop_idx % inference_interval == 0)

            frames = camera.get_frames()
            if frames is None:
                time.sleep(0.01)
                continue

            color_image, depth_image = frames

            # --- YOLO inference (every N frames) -----------------------
            if run_yolo:
                raw_dets = detector.detect(color_image)
                last_detections = []
                for det in raw_dets:
                    xmin, ymin, xmax, ymax = map(int, det["bbox"])
                    cx = int((xmin + xmax) / 2)
                    cy = int((ymin + ymax) / 2)

                    # Robust depth at detection centre
                    dist = camera.get_distance_at_pixel(depth_image, cx, cy)
                    if dist is None:
                        continue  # skip detections with no depth

                    # Project pixel → camera-frame 3D coords (metres)
                    rx = (cx - camera.ppx) * dist / camera.fx
                    ry = (cy - camera.ppy) * dist / camera.fy
                    rz = dist

                    last_detections.append({
                        **det,
                        "cx": cx, "cy": cy,
                        "dist": dist,
                        "rx": rx, "ry": ry, "rz": rz,
                    })

                # Feed ALL detections to multi-target tracker in one batch
                tracker.update(last_detections, dt=1.0 / 30.0)
            else:
                # Between YOLO frames, propagate the Kalman prediction only
                tracker.predict(dt=1.0 / 30.0)

            # --- FPS counter -------------------------------------------
            fps_frame_count += 1
            now = time.monotonic()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer = now

            # --- Fetch Telemetry and Log at 10Hz -----------------------
            from mavlink_interface.connection import TelemetrySnapshot
            if conn and conn.is_connected():
                telemetry = conn.get_telemetry()
            else:
                telemetry = TelemetrySnapshot(connected=False)

            target_vector = tracker.get_primary_target()
            trk_status = "TRACKING" if tracker.is_tracking else "SEARCHING"
            n_active_tracks = len(tracker.confirmed_tracks)

            if now - last_log_time >= (1.0 / 10.0):
                logger.log(
                    state_name=trk_status,
                    telemetry=telemetry,
                    target_vector=target_vector,
                    cmd_vector=None,
                )
                last_log_time = now

            # --- Build Annotated Frame for recording/display ----------
            if (video_writer and video_writer.isOpened()) or show_display:
                annotated = color_image.copy()

                for det in last_detections:
                    xmin, ymin, xmax, ymax = det["bbox"]
                    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                    cx, cy = det["cx"], det["cy"]
                    conf = det["confidence"]
                    dist = det["dist"]
                    cls_id = det["class_id"]

                    # Green = drone (class 1), Red = balloon (class 0)
                    color = (0, 255, 0) if cls_id == 1 else (0, 0, 255)
                    label_name = "Drone" if cls_id == 1 else "Balloon"

                    cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
                    label = f"{label_name} {conf:.2f} {dist:.2f}m"
                    (w_l, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (xmin, ymin - 20), (xmin + w_l, ymin), color, -1)
                    cv2.putText(annotated, label, (xmin, ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)

                # Kalman tracker overlay — draw ALL confirmed tracks
                all_track_states = tracker.get_all_states()
                primary_state = tracker.get_primary_target()
                for tinfo in all_track_states:
                    tstate = tinfo["state"]
                    t_id = tinfo["track_id"]
                    t_cls = tinfo["class_id"]
                    px, py, pz, vx, vy, vz = tstate
                    if pz > 0:
                        proj_x = int(px * camera.fx / pz + camera.ppx)
                        proj_y = int(py * camera.fy / pz + camera.ppy)
                        if 0 <= proj_x < 640 and 0 <= proj_y < 480:
                            is_primary = (primary_state is not None and
                                          abs(tstate[0] - primary_state[0]) < 1e-6 and
                                          abs(tstate[2] - primary_state[2]) < 1e-6)
                            clr = (0, 255, 255) if is_primary else (255, 200, 0)
                            sz = 20 if is_primary else 14
                            cv2.drawMarker(annotated, (proj_x, proj_y),
                                           clr, cv2.MARKER_CROSS, sz, 2)
                            cls_tag = "B" if t_cls == 0 else "D"
                            kf_label = f"T{t_id}({cls_tag}) {pz:.1f}m v={vz:.1f}"
                            cv2.putText(annotated, kf_label, (proj_x + 10, proj_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, clr, 1, cv2.LINE_AA)

                # FPS + tracker status overlay
                cv2.putText(
                    annotated,
                    f"FPS: {fps_display:.1f}  Interval: {inference_interval}x  Tracks: {n_active_tracks}  [{trk_status}]",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                )

                # Write to video
                if video_writer and video_writer.isOpened():
                    video_writer.write(annotated)

                # Show display
                if show_display:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except cv2.error:
                        break

            # --- Throttling in mock mode ------------------------------
            if camera.is_mock:
                elapsed = time.monotonic() - t_loop_start
                sleep_time = max(0.001, (1.0 / 15.0) - elapsed)  # target 15 FPS
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down.")
    finally:
        log.info("Stopping subsystems...")
        if logger:
            logger.stop()
        if conn:
            conn.close()
        if video_writer:
            video_writer.release()
            log.info("Video saved → %s", rec_path)
        camera.stop()
        if show_display:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        log.info("Formyx Backend stopped cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

