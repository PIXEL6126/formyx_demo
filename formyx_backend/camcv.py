"""
formyx_backend/camcv.py
-----------------------
Standalone dual-class detection viewer + recorder with Intel RealSense D435i depth camera.

Displays a live OpenCV window with balloon (red) and drone (green) detection
overlays, FPS counter, metric depth measurement, and 3D Kalman-tracker crosshair.
Every session is saved as a timestamped .mp4 in the camrec/ folder.

NOTE: This version uses the Intel RealSense D435i depth camera connected over
      USB 3.0 ports. Aligned RGB + Depth streams are acquired using pyrealsense2.

Usage
-----
    cd formyx_backend
    DISPLAY=:0 python3 camcv.py

Options
-------
    --mock              Run in mock mode (simulated RealSense camera)
    --width INT         Capture width in pixels (default: 640)
    --height INT        Capture height in pixels (default: 480)
    --conf FLOAT        Detection confidence threshold (default: 0.25)
    --interval INT      Run YOLO every N frames (default: 2, raise to 3 for
                        slower Pi boards)
    --no-tracker        Disable the Kalman tracker overlay
    --no-tiled          Disable tiled (SAHI) inference (faster but worse
                        long-range detection)
    --no-record         Stream only, do not save video
    --no-display        Headless mode — record without showing a window
    --output-dir PATH   Override recording directory (default: camrec)
    --segment-mins INT  Split recording into segments of this length in minutes.
                        When the segment ends the current .mp4 is saved and a
                        new one starts automatically. (default: 3, 0 = disabled)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from depth.realsense_interface import RealSenseInterface
from perception.detector import ObjectDetector
from tracking.multi_target_tracker import MultiTargetTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("camcv")

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
CLR_BALLOON  = (0,   0, 255)   # Red    — balloon
CLR_DRONE    = (0, 220,   0)   # Green  — drone
CLR_KF       = (0, 255, 255)   # Yellow — Kalman projection (primary target)
CLR_KF_SEC   = (255, 200, 0)   # Cyan   — secondary tracked targets
CLR_FPS      = (0, 255,   0)   # Green  — HUD text
CLR_WHITE    = (255, 255, 255)
CLR_BLACK    = (0,   0,   0)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="camcv",
        description="Formyx dual-class detection viewer + recorder (Intel RealSense D435i depth camera)",
    )
    p.add_argument("--mock",       action="store_true",
                   help="Run in mock mode with simulated RealSense camera")
    p.add_argument("--width",      type=int,   default=640,
                   help="Capture width in pixels (default 640)")
    p.add_argument("--height",     type=int,   default=480,
                   help="Capture height in pixels (default 480)")
    p.add_argument("--conf",       type=float, default=0.25,
                   help="Detection confidence threshold (default 0.25)")
    p.add_argument("--interval",   type=int,   default=2,
                   help="Run YOLO every N frames (default 2)")
    p.add_argument("--no-tracker", action="store_true",
                   help="Disable the Kalman tracker overlay")
    p.add_argument("--no-tiled",   action="store_true",
                   help="Disable tiled/SAHI inference (faster, less long-range accuracy)")
    p.add_argument("--no-record",  action="store_true",
                   help="Do not save a video recording")
    p.add_argument("--no-display", action="store_true",
                   help="Disable the OpenCV camera window (headless mode)")
    p.add_argument("--output-dir", default="camrec",
                   help="Recording output directory (default: camrec/)")
    p.add_argument("--segment-mins", type=int, default=3,
                   help="Split recording every N minutes (default 3, 0 = disabled)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_box(img: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int,
              color: tuple, label: str) -> None:
    """Draw a bounding box with a filled label tab above it."""
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(img, (xmin, ymin - 20), (xmin + tw + 4, ymin), color, -1)
    cv2.putText(img, label, (xmin + 2, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1, cv2.LINE_AA)


def _draw_hud(img: np.ndarray, fps: float, interval: int,
              n_balloon: int, n_drone: int, n_tracks: int,
              tracking: bool) -> None:
    """Render the on-screen HUD (top-left corner)."""
    lines = [
        f"FPS: {fps:.1f}  |  interval: {interval}x",
        f"Balloons: {n_balloon}   Drones: {n_drone}   Tracks: {n_tracks}",
        f"Tracker: {'LOCKED' if tracking else 'SEARCHING'}",
    ]
    y = 28
    for line in lines:
        # Drop-shadow for readability
        cv2.putText(img, line, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_BLACK, 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_FPS, 2, cv2.LINE_AA)
        y += 26


def _draw_kalman_tracks(img: np.ndarray, tracker: MultiTargetTracker,
                        camera: RealSenseInterface) -> None:
    """
    Overlay ALL confirmed 3D Kalman tracks using RealSense camera intrinsics.
    Primary target → yellow crosshair, secondary → cyan markers.
    """
    all_states = tracker.get_all_states()
    primary = tracker.get_primary_target()

    h, w = img.shape[:2]

    for info in all_states:
        state = info["state"]
        track_id = info["track_id"]
        cls_id = info["class_id"]
        px, py, pz, vx, vy, vz = state

        if pz <= 0.05:
            continue

        proj_x = int(px * camera.fx / pz + camera.ppx)
        proj_y = int(py * camera.fy / pz + camera.ppy)

        if not (0 <= proj_x < w and 0 <= proj_y < h):
            continue

        # Determine if this is the primary target
        is_primary = (primary is not None and
                      abs(state[0] - primary[0]) < 1e-6 and
                      abs(state[2] - primary[2]) < 1e-6)

        color = CLR_KF if is_primary else CLR_KF_SEC
        marker_sz = 22 if is_primary else 16

        cv2.drawMarker(img, (proj_x, proj_y), color,
                       cv2.MARKER_CROSS, markerSize=marker_sz, thickness=2)

        cls_tag = "B" if cls_id == 0 else "D"
        label = f"T{track_id}({cls_tag}) {pz:.1f}m v=({vx:+.1f},{vy:+.1f})"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        bx = min(proj_x + 12, w - tw - 4)
        cv2.rectangle(img, (bx - 2, proj_y - 14), (bx + tw + 2, proj_y + 2),
                      CLR_BLACK, -1)
        cv2.putText(img, label, (bx, proj_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Video writer setup
# ---------------------------------------------------------------------------

def _make_writer(output_dir: str, width: int, height: int) -> tuple:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"session_{stamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 15.0, (width, height))
    if not writer.isOpened():
        log.warning("VideoWriter failed to open — recording disabled.")
        return None, out_path
    log.info("Recording to: %s", out_path)
    return writer, out_path


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------
_shutdown_requested = False

def _handle_signal(signum, frame):  # noqa: ANN001
    global _shutdown_requested
    logging.getLogger("camcv").warning(
        "Signal %d received — initiating clean shutdown.", signum)
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse()
    show_display = not args.no_display

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ------------------------------------------------------------------
    # 1. Open Intel RealSense D435i camera
    # ------------------------------------------------------------------
    log.info("Initialising Intel RealSense D435i depth camera (USB 3.0)…")
    camera = RealSenseInterface(use_mock=args.mock)
    camera.start()
    if camera.is_mock:
        log.warning("RealSense interface is running in MOCK mode.")
    else:
        log.info(
            "RealSense camera ready — intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
            camera.fx, camera.fy, camera.ppx, camera.ppy,
        )

    W, H = 640, 480

    # ------------------------------------------------------------------
    # 2. Detector
    # ------------------------------------------------------------------
    log.info("Loading dual-class YOLO detector (balloon + drone)…")
    enable_tiled = not getattr(args, 'no_tiled', False)
    detector = ObjectDetector(enable_tiled=enable_tiled)
    log.info("Detector ready (ONNX=%s, tiled=%s).", detector.use_onnx, enable_tiled)

    # ------------------------------------------------------------------
    # 3. Multi-target tracker (optional)
    # ------------------------------------------------------------------
    tracker = MultiTargetTracker() if not args.no_tracker else None

    # ------------------------------------------------------------------
    # 4. Video writer
    # ------------------------------------------------------------------
    writer, rec_path = None, None
    if not args.no_record:
        writer, rec_path = _make_writer(args.output_dir, W, H)

    # ------------------------------------------------------------------
    # 5. Display window  (skipped when --no-display)
    # ------------------------------------------------------------------
    WINDOW = "Formyx Detection — Q / Esc to quit"
    if show_display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, W, H)
        log.info("Press Q or Esc in the window to stop.")
    else:
        log.info("Running headless (no display). Press Ctrl-C to stop.")

    # ------------------------------------------------------------------
    # 6. Main loop
    # ------------------------------------------------------------------
    loop_idx     = 0
    fps_display  = 0.0
    fps_count    = 0
    fps_t0       = time.monotonic()
    last_dets: list = []
    segment_secs = args.segment_mins * 60 if args.segment_mins > 0 else None
    seg_start_t  = time.monotonic()   # tracks when current segment started

    try:
        while not _shutdown_requested:
            t_start = time.monotonic()
            loop_idx += 1
            run_yolo = (loop_idx % args.interval == 0)

            # --- grab frame -------------------------------------------
            frames = camera.get_frames()
            if frames is None:
                log.warning("Failed to grab frames from RealSense — retrying…")
                time.sleep(0.05)
                continue

            color, depth = frames

            # --- YOLO inference ---------------------------------------
            n_balloon = n_drone = 0
            if run_yolo:
                raw = detector.detect(color)
                last_dets = []
                for det in raw:
                    xmin, ymin, xmax, ymax = map(int, det["bbox"])
                    cx = int((xmin + xmax) / 2)
                    cy = int((ymin + ymax) / 2)

                    # Query robust metric distance at pixel centre
                    dist = camera.get_distance_at_pixel(depth, cx, cy)

                    rx, ry, rz = None, None, None
                    if dist is not None:
                        # Project 2D pixel + depth → 3D camera-frame coordinates (metres)
                        rx = (cx - camera.ppx) * dist / camera.fx
                        ry = (cy - camera.ppy) * dist / camera.fy
                        rz = dist

                    last_dets.append({
                        **det,
                        "cx": cx, "cy": cy,
                        "dist": dist,
                        "rx": rx, "ry": ry, "rz": rz,
                    })

                # Feed ALL detections to multi-target tracker in one batch
                if tracker:
                    tracker.update(last_dets, dt=1.0 / 30.0)
            elif tracker:
                tracker.predict(dt=1.0 / 30.0)

            # --- FPS counter ------------------------------------------
            fps_count += 1
            now = time.monotonic()
            if now - fps_t0 >= 1.0:
                fps_display = fps_count / (now - fps_t0)
                fps_count   = 0
                fps_t0      = now
                n_b = sum(1 for d in last_dets if d["class_id"] == 0)
                n_d = sum(1 for d in last_dets if d["class_id"] == 1)
                log.info(
                    "FPS=%.1f  Balloons=%d  Drones=%d  Tracker=%s",
                    fps_display, n_b, n_d,
                    "LOCKED" if (tracker and tracker.is_initialized) else "SEARCHING",
                )

            # --- Annotate frame ---------------------------------------
            need_annotate = (writer and writer.isOpened()) or show_display
            if need_annotate:
                vis = color.copy()
                n_b = n_d = 0
                for det in last_dets:
                    xmin, ymin, xmax, ymax = map(int, det["bbox"])
                    cx, cy  = det["cx"], det["cy"]
                    conf    = det["confidence"]
                    dist    = det["dist"]
                    cls_id  = det["class_id"]
                    color_box = CLR_BALLOON if cls_id == 0 else CLR_DRONE
                    name      = "Balloon"   if cls_id == 0 else "Drone"
                    if cls_id == 0:
                        n_b += 1
                    else:
                        n_d += 1

                    dist_str = f"{dist:.2f}m" if dist is not None else "N/A"
                    _draw_box(vis, xmin, ymin, xmax, ymax, color_box,
                              f"{name}  {conf:.2f}  {dist_str}")
                    cv2.circle(vis, (cx, cy), 4, CLR_KF, -1)

                if tracker:
                    _draw_kalman_tracks(vis, tracker, camera)

                n_tracks = len(tracker.confirmed_tracks) if tracker else 0
                _draw_hud(vis, fps_display, args.interval, n_b, n_d, n_tracks,
                          tracker.is_tracking if tracker else False)

                # --- Write to recording ---------------------------
                if writer and writer.isOpened():
                    writer.write(vis)

            # ----------------------------------------------------------
            # Segment rotation — close current file, open a new one
            # ----------------------------------------------------------
            if segment_secs and writer and (time.monotonic() - seg_start_t >= segment_secs):
                log.info("Segment limit reached — rotating video file…")
                writer.release()
                log.info("Segment saved → %s", rec_path)
                writer, rec_path = _make_writer(args.output_dir, W, H)
                seg_start_t = time.monotonic()

            # --- Show window (only when not headless) ---------
            if show_display:
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

            # Throttling in mock mode to avoid tight loops
            if camera.is_mock:
                time.sleep(0.01)

    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        log.info("Shutting down…")
        if writer and writer.isOpened():
            writer.release()
            log.info("Video saved → %s", rec_path)
        camera.stop()
        if show_display:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        log.info("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
