"""
tools/live_viewer.py
--------------------
Live colour + depth side-by-side viewer for the Intel RealSense D435i.
Run this in a background terminal while hardware tests are in progress.

Usage:
    python3 tools/live_viewer.py

Keys:
    Q  — quit
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,  30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,   30)

    align   = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)  # cool-warm

    print("Starting RealSense pipeline...")
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        sys.exit(1)

    print("Viewer running — press Q in the window to quit.\n")

    cv2.namedWindow("Formyx — RealSense Live View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Formyx — RealSense Live View", 1280, 520)

    fps_display = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frames        = pipeline.wait_for_frames(timeout_ms=5000)
            aligned       = align.process(frames)
            color_frame   = aligned.get_color_frame()
            depth_frame   = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img  = np.asanyarray(color_frame.get_data())
            depth_coloured = np.asanyarray(
                colorizer.colorize(depth_frame).get_data()
            )

            # Calculate FPS
            frame_count += 1
            curr_time = time.time()
            elapsed = curr_time - start_time
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                start_time = curr_time

            # Overlay FPS counter on color image
            fps_label = f"FPS: {fps_display:.1f}"
            cv2.putText(color_img, fps_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Overlay centre-pixel distance on depth frame
            dist = depth_frame.get_distance(320, 240)
            label = f"Centre: {dist:.3f} m" if dist > 0 else "Centre: no depth"
            cv2.putText(depth_coloured, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw crosshair at centre
            for img in (color_img, depth_coloured):
                cv2.drawMarker(img, (320, 240), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2)

            # Side-by-side
            combined = np.hstack([color_img, depth_coloured])
            cv2.imshow("Formyx — RealSense Live View", combined)

            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Viewer closed.")

if __name__ == "__main__":
    main()
