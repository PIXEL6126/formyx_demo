"""
formyx_backend/main.py
-----------------------
Demonstration-Ready Top-Level Entry Point for Formyx Autonomous Drone System.

Features:
  • Multi-Threaded Architecture: 30 FPS smooth display loop + async ONNX background inference worker.
  • Aspect-Ratio Preserving Letterbox Preprocessing & Soft-NMS (high confidence detection).
  • Intel RealSense D435i aligned RGB + Depth acquisition with robust depth-shadow patch averaging.
  • Dual-Class Detection (Balloon + Drone) with metric distance & 3D camera coordinates.
  • 3D Kalman Multi-Target Tracker with target locking and velocity estimation.
  • Demonstration HUD: Sleek overlay, Depth Picture-in-Picture (PIP), and 1-key screenshot capture.
  • Optional MAVLink Telemetry (PIX6 flight controller connection or fallback).
  • BlackBox Flight Logger & Automatic Video Recorder (.mp4).

Usage:
  cd formyx_backend
  python3 main.py [OPTIONS]

Options:
  --no-mavlink          Skip MAVLink flight controller connection (camera + perception only)
  --connection STR      MAVLink connection string (overrides settings.yaml)
  --conf FLOAT          Detection confidence threshold (default: 0.20)
  --no-display          Headless mode (disable OpenCV display window)
  --no-record           Disable saving video recording to camrec/
  --no-tiled            Disable tiled/SAHI inference (faster but less long-range accuracy)
  --depth-pip           Show Depth Map Picture-in-Picture in bottom-right corner by default
  --output-dir PATH     Directory for recordings and screenshots (default: camrec)

Interactive Keys in Display Window:
  Q / Esc : Exit application cleanly
  D       : Toggle Depth Map Picture-in-Picture (PIP) view
  T       : Toggle 3D Kalman Filter tracking overlay
  S       : Take a high-resolution screenshot snapshot
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import load_config, get
from depth.realsense_interface import RealSenseInterface
from perception.detector import ObjectDetector
from tracking.multi_target_tracker import MultiTargetTracker
from logging_system.logger import BlackBoxLogger

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Colors (BGR)
# ---------------------------------------------------------------------------
CLR_BALLOON = (0, 0, 255)       # Red
CLR_DRONE   = (0, 220, 0)       # Green
CLR_KF_PRI  = (0, 255, 255)     # Bright Yellow (Primary Lock)
CLR_KF_SEC  = (255, 200, 0)     # Cyan (Secondary Track)
CLR_HUD_BG  = (20, 20, 20)      # Dark translucent background
CLR_TEXT    = (255, 255, 255)   # White
CLR_GREEN   = (0, 255, 0)       # Success Green
CLR_ORANGE  = (0, 165, 255)     # Medium Conf
CLR_SHADOW  = (0, 0, 0)         # Shadow for readable text

# ---------------------------------------------------------------------------
# Signal Handling
# ---------------------------------------------------------------------------
_stop_event = threading.Event()

def _handle_signal(signum, frame):  # noqa: ANN001
    log.warning("Signal %d received — requesting clean shutdown.", signum)
    _stop_event.set()

# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Formyx Autonomous Balloon-Tracking Drone — Demonstration Engine",
    )
    parser.add_argument("--connection", default=None,
                        help="MAVLink connection string (e.g. serial:/dev/ttyAMA0:921600 or udpin:localhost:14550)")
    parser.add_argument("--no-mavlink", action="store_true",
                        help="Skip MAVLink flight controller connection")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Detection confidence threshold (default: 0.20)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV camera display window")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable video recording")
    parser.add_argument("--no-tiled", action="store_true",
                        help="Disable tiled/SAHI inference")
    parser.add_argument("--depth-pip", action="store_true",
                        help="Enable Depth Picture-in-Picture on start")
    parser.add_argument("--output-dir", default="camrec",
                        help="Directory for output videos and snapshots (default: camrec)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Video Writer
# ---------------------------------------------------------------------------
def _make_writer(output_dir: str, width: int = 640, height: int = 480) -> tuple:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"session_{stamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 30.0, (width, height))
    if not writer.isOpened():
        log.warning("VideoWriter failed to open — recording disabled.")
        return None, out_path
    log.info("Recording live demonstration to: %s", out_path)
    return writer, out_path

# ---------------------------------------------------------------------------
# Background Inference Worker Thread
# ---------------------------------------------------------------------------
def _inference_worker(detector: ObjectDetector, camera: RealSenseInterface,
                      tracker: MultiTargetTracker, shared_state: dict):
    log.info("Background AI inference worker thread started.")
    last_run_time = time.monotonic()
    
    while not _stop_event.is_set():
        with shared_state["lock"]:
            frame = shared_state["input_frame"]
            depth_frame = shared_state["input_depth"]
            shared_state["input_frame"] = None  # Consume frame

        if frame is None:
            time.sleep(0.005)
            continue

        t0 = time.monotonic()

        # Run ONNX YOLO detector (includes letterbox preprocessing & Soft-NMS)
        raw_dets = detector.detect(frame)

        processed_dets = []
        for det in raw_dets:
            xmin, ymin, xmax, ymax = map(int, det["bbox"])
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            # Query spatial depth distance
            dist = camera.get_distance_at_pixel(depth_frame, cx, cy)
            if dist is None:
                continue

            rx = (cx - camera.ppx) * dist / camera.fx
            ry = (cy - camera.ppy) * dist / camera.fy
            rz = dist

            processed_dets.append({
                **det,
                "cx": cx, "cy": cy,
                "dist": dist,
                "rx": rx, "ry": ry, "rz": rz,
            })

        now = time.monotonic()
        dt = max(0.001, now - last_run_time)
        last_run_time = now

        # Update 3D Kalman multi-target tracker
        tracker.update(processed_dets, dt=dt)

        inf_ms = (time.monotonic() - t0) * 1000.0

        with shared_state["lock"]:
            shared_state["detections"] = processed_dets
            shared_state["inference_ms"] = inf_ms

# ---------------------------------------------------------------------------
# HUD Rendering Functions
# ---------------------------------------------------------------------------
def _draw_text_shadow(img: np.ndarray, text: str, pos: tuple,
                      font_scale: float = 0.5, color: tuple = (255, 255, 255),
                      thickness: int = 1):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, CLR_SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def _draw_hud_banner(img: np.ndarray, fps_disp: float, fps_ai: float,
                     n_balloon: int, n_drone: int, n_tracks: int,
                     is_tracking: bool, target_info: str,
                     telem_summary: str):
    h, w = img.shape[:2]
    
    # Semi-transparent top bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), CLR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.line(img, (0, 54), (w, 54), (0, 255, 0) if is_tracking else (100, 100, 100), 1)

    # Line 1: Main Status & FPS
    status_str = "FORMYX DEMO ENGINE"
    fps_str = f"Display: {fps_disp:.1f} FPS | AI: {fps_ai:.1f} FPS"
    counts_str = f"Balloons: {n_balloon}  Drones: {n_drone}  3D Tracks: {n_tracks}"
    
    _draw_text_shadow(img, f"[{status_str}]  {fps_str}", (10, 22), 0.52, (0, 255, 255), 2)
    _draw_text_shadow(img, counts_str, (w - 320, 22), 0.50, CLR_TEXT, 1)

    # Line 2: Tracker Lock & Telemetry Status
    lock_color = CLR_GREEN if is_tracking else CLR_ORANGE
    _draw_text_shadow(img, f"Tracker: {target_info}", (10, 44), 0.48, lock_color, 2)
    if telem_summary:
        _draw_text_shadow(img, telem_summary, (w - 380, 44), 0.42, (200, 200, 200), 1)

def _draw_bounding_boxes(img: np.ndarray, detections: list):
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cx, cy = det["cx"], det["cy"]
        conf = det["confidence"]
        dist = det["dist"]
        cls_id = det["class_id"]

        color = CLR_BALLOON if cls_id == 0 else CLR_DRONE
        label_name = "Balloon" if cls_id == 0 else "Drone"

        # Bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        # Label tab
        label = f"{label_name} {conf:.2f} {dist:.2f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + tw + 6, ymin), color, -1)
        cv2.putText(img, label, (xmin + 3, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Center point
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)

def _draw_kalman_projections(img: np.ndarray, tracker: MultiTargetTracker,
                             camera: RealSenseInterface):
    h, w = img.shape[:2]
    all_tracks = tracker.get_all_states()
    primary_target = tracker.get_primary_target()

    for info in all_tracks:
        state = info["state"]
        t_id = info["track_id"]
        cls_id = info["class_id"]
        px, py, pz, vx, vy, vz = state

        if pz <= 0.05:
            continue

        proj_x = int(px * camera.fx / pz + camera.ppx)
        proj_y = int(py * camera.fy / pz + camera.ppy)

        if not (0 <= proj_x < w and 0 <= proj_y < h):
            continue

        is_primary = (primary_target is not None and
                      abs(state[0] - primary_target[0]) < 1e-5 and
                      abs(state[2] - primary_target[2]) < 1e-5)

        color = CLR_KF_PRI if is_primary else CLR_KF_SEC
        sz = 22 if is_primary else 15

        # Draw crosshair marker
        cv2.drawMarker(img, (proj_x, proj_y), color, cv2.MARKER_CROSS, sz, 2)
        if is_primary:
            cv2.circle(img, (proj_x, proj_y), sz // 2, color, 1)

        # Velocity label
        cls_tag = "B" if cls_id == 0 else "D"
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        lbl = f"T{t_id}({cls_tag}) {pz:.1f}m v={v_mag:.1f}m/s"
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        
        bx = min(proj_x + 12, w - tw - 4)
        cv2.rectangle(img, (bx - 2, proj_y - 14), (bx + tw + 2, proj_y + 2), (0, 0, 0), -1)
        cv2.putText(img, lbl, (bx, proj_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

def _render_depth_pip(img: np.ndarray, depth_img: np.ndarray):
    """Render a colorized Depth Map Picture-in-Picture in bottom-right corner."""
    h, w = img.shape[:2]
    pip_w, pip_h = 180, 135

    if depth_img is None:
        return

    # Convert depth image to colorized visualization
    if np.issubdtype(depth_img.dtype, np.integer):
        depth_scaled = cv2.convertScaleAbs(depth_img, alpha=0.03)
    else:
        depth_scaled = cv2.convertScaleAbs(depth_img * 1000.0, alpha=0.03)

    color_depth = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
    pip_resized = cv2.resize(color_depth, (pip_w, pip_h))

    # Paste into bottom right with border
    x1, y1 = w - pip_w - 12, h - pip_h - 12
    x2, y2 = w - 12, h - 12

    img[y1:y2, x1:x2] = pip_resized
    cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 255, 255), 1)
    _draw_text_shadow(img, "DEPTH PIP", (x1 + 6, y1 + 16), 0.38, (255, 255, 255), 1)

# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------
def main() -> int:
    args = _parse_args()
    log.info("Starting Formyx Demonstration Engine...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cfg = load_config()

    # 1. MAVLink Telemetry Connection (Optional)
    conn = None
    if not args.no_mavlink:
        conn_string = args.connection or cfg.get("mavlink", {}).get("connection_string", "/dev/ttyACM0,57600")
        log.info("Attempting MAVLink connection: %s", conn_string)
        try:
            from mavlink_interface.connection import MAVLinkConnection
            conn = MAVLinkConnection(connection_string=conn_string)
            conn.connect()
            log.info("[+] MAVLink telemetry active.")
        except Exception as exc:
            log.warning("[-] MAVLink connection skipped: %s. Operating in Standalone Perception Mode.", exc)
            conn = None

    # 2. Intel RealSense D435i Camera
    log.info("Initialising Intel RealSense D435i Depth Camera...")
    camera = RealSenseInterface(use_mock=False)
    camera.start()

    # 3. Dual-Class ONNX YOLO Object Detector (Letterbox + Soft-NMS)
    log.info("Loading Optimized Dual-Class YOLO Detector...")
    enable_tiled = not args.no_tiled
    detector = ObjectDetector(enable_tiled=enable_tiled)
    log.info("[+] Detector ready (ONNX=%s, Tiled=%s, Conf=%.2f)", detector.use_onnx, enable_tiled, args.conf)

    # 4. Multi-Target 3D Kalman Tracker
    tracker = MultiTargetTracker()

    # 5. BlackBox Flight Data Logger
    logger = BlackBoxLogger()
    logger.start()
    last_log_time = time.monotonic()

    # 6. Video Recorder
    video_writer, rec_path = None, None
    if not args.no_record:
        video_writer, rec_path = _make_writer(args.output_dir, 640, 480)

    # 7. Display Window Setup
    show_display = not args.no_display
    window_name = "Formyx Autonomous Drone — Live Demonstration"
    if show_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

    # 8. Threaded State Initialization
    shared_state = {
        "lock": threading.Lock(),
        "input_frame": None,
        "input_depth": None,
        "detections": [],
        "inference_ms": 0.0,
    }

    worker = threading.Thread(
        target=_inference_worker,
        args=(detector, camera, tracker, shared_state),
        daemon=True,
    )
    worker.start()

    # 9. Main 30 FPS Render & Control Loop
    show_depth_pip = args.depth_pip
    show_kalman = True
    notification_txt = ""
    notification_until = 0.0

    fps_display = 0.0
    frame_count = 0
    fps_start_time = time.monotonic()

    log.info("\n=====================================================================")
    log.info("  FORMYX LIVE DEMONSTRATION ENGINE ACTIVE")
    log.info("    - Display Loop : 30 FPS smooth continuous video feed")
    log.info("    - AI Engine    : Multi-threaded ONNX background worker")
    log.info("    - Preprocess   : Letterbox aspect-ratio preservation")
    log.info("    - Postprocess  : Gaussian Soft-NMS score decay")
    log.info("    - Controls     : Q=Quit | D=Depth PIP | T=Tracker | S=Snapshot")
    log.info("=====================================================================\n")

    try:
        while not _stop_event.is_set():
            t_loop_start = time.monotonic()

            frames = camera.get_frames()
            if frames is None:
                time.sleep(0.005)
                continue

            color_img, depth_img = frames

            # Pass latest frame to background AI thread
            with shared_state["lock"]:
                shared_state["input_frame"] = color_img.copy()
                shared_state["input_depth"] = depth_img
                current_dets = shared_state["detections"]
                inf_ms = shared_state["inference_ms"]

            inf_fps = 1000.0 / inf_ms if inf_ms > 0 else 0.0

            # Calculate main display FPS
            frame_count += 1
            now = time.monotonic()
            elapsed = now - fps_start_time
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_start_time = now

            # Fetch Telemetry & Log at 10Hz
            from mavlink_interface.connection import TelemetrySnapshot
            telemetry = conn.get_telemetry() if (conn and conn.is_connected()) else TelemetrySnapshot(connected=False)
            
            primary_target = tracker.get_primary_target()
            is_tracking = tracker.is_tracking
            n_tracks = len(tracker.confirmed_tracks)

            if now - last_log_time >= 0.1:
                logger.log(
                    state_name="TRACKING" if is_tracking else "SEARCHING",
                    telemetry=telemetry,
                    target_vector=primary_target,
                    cmd_vector=None,
                )
                last_log_time = now

            # Render Frame
            if (video_writer and video_writer.isOpened()) or show_display:
                annotated = color_img.copy()

                # 1. Draw Detections
                _draw_bounding_boxes(annotated, current_dets)

                # 2. Draw 3D Kalman Tracker Projections
                if show_kalman:
                    _draw_kalman_projections(annotated, tracker, camera)

                # 3. Draw Depth Picture-in-Picture
                if show_depth_pip:
                    _render_depth_pip(annotated, depth_img)

                # 4. Count Targets
                n_b = sum(1 for d in current_dets if d["class_id"] == 0)
                n_d = sum(1 for d in current_dets if d["class_id"] == 1)

                # Build Target Info String
                if primary_target is not None:
                    target_str = f"LOCKED — dist={primary_target[2]:.2f}m"
                elif tracker.is_initialized:
                    target_str = "SEARCHING (filter active)"
                else:
                    target_str = "SEARCHING"

                # Telemetry Summary String
                telem_summary = ""
                if telemetry.connected:
                    telem_summary = f"Armed:{telemetry.armed} | {telemetry.flight_mode} | Alt:{telemetry.alt_agl_m:.1f}m | Bat:{telemetry.battery_remaining_pct}%"

                # 5. Draw Top HUD Banner
                _draw_hud_banner(annotated, fps_display, inf_fps,
                                 n_b, n_d, n_tracks, is_tracking,
                                 target_str, telem_summary)

                # Notification Banner (e.g. Snapshot Saved)
                if now < notification_until:
                    _draw_text_shadow(annotated, notification_txt, (10, 75), 0.60, (0, 255, 0), 2)

                # Write Video Frame
                if video_writer and video_writer.isOpened():
                    video_writer.write(annotated)

                # Display OpenCV Window
                if show_display:
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), ord("Q"), 27):
                        break
                    elif key in (ord("d"), ord("D")):
                        show_depth_pip = not show_depth_pip
                        log.info("Toggled Depth PIP: %s", show_depth_pip)
                    elif key in (ord("t"), ord("T")):
                        show_kalman = not show_kalman
                        log.info("Toggled Kalman Overlay: %s", show_kalman)
                    elif key in (ord("s"), ord("S")):
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snap_path = os.path.join(args.output_dir, f"snapshot_{stamp}.jpg")
                        cv2.imwrite(snap_path, annotated)
                        notification_txt = f"[+] Snapshot saved: {snap_path}"
                        notification_until = now + 2.5
                        log.info(notification_txt)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        log.info("Shutting down demonstration engine...")
        _stop_event.set()
        if logger:
            logger.stop()
        if conn:
            conn.close()
        if video_writer:
            video_writer.release()
            log.info("[+] Session video saved → %s", rec_path)
        camera.stop()
        if show_display:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        log.info("[+] Clean shutdown complete.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
