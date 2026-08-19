#!/usr/bin/env python3
"""
tools/test_realsense.py
-----------------------
Quick sanity check: is the Intel RealSense D435i alive on USB 3.0?

Prints camera info, grabs 10 frames, and reports resolution + FPS.
No display window needed — runs headless.

Usage:
    python3 tools/test_realsense.py
"""

import sys
import time

try:
    import pyrealsense2 as rs
except ImportError:
    print("✗  pyrealsense2 is NOT installed.")
    sys.exit(1)

print("✓  pyrealsense2 imported OK")

# ── Discover connected devices ──────────────────────────────────────
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("✗  No RealSense devices found. Check USB 3.0 connection.")
    sys.exit(1)

for i, dev in enumerate(devices):
    print(f"\n── Device {i} ──")
    print(f"   Name     : {dev.get_info(rs.camera_info.name)}")
    print(f"   Serial   : {dev.get_info(rs.camera_info.serial_number)}")
    print(f"   Firmware : {dev.get_info(rs.camera_info.firmware_version)}")
    try:
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"   USB Type : {usb_type}")
        if usb_type.startswith("2."):
            print("   ⚠  Camera is on a USB 2.0 port — use USB 3.0 for depth!")
    except Exception:
        pass

# ── Start pipeline ──────────────────────────────────────────────────
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

print("\nStarting pipeline (color 640×480 + depth 640×480 @ 30fps)…")
try:
    profile = pipeline.start(config)
except Exception as e:
    print(f"✗  Failed to start pipeline: {e}")
    sys.exit(1)

print("✓  Pipeline started")

# Print intrinsics
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
print(f"   Intrinsics: fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
      f"cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"   Depth scale: {depth_scale:.6f} m/unit")

# ── Grab frames ─────────────────────────────────────────────────────
NUM_FRAMES = 30
print(f"\nGrabbing {NUM_FRAMES} frames…")
t0 = time.monotonic()
good = 0

for i in range(NUM_FRAMES):
    try:
        frames = pipeline.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if color and depth:
            good += 1
            # Print centre-pixel depth on first + last frame
            if i == 0 or i == NUM_FRAMES - 1:
                d = depth.get_distance(320, 240)
                print(f"   Frame {i+1:3d}:  color {color.get_width()}×{color.get_height()}  "
                      f"depth {depth.get_width()}×{depth.get_height()}  "
                      f"centre={d:.3f}m")
    except Exception as e:
        print(f"   Frame {i+1:3d}:  ✗ {e}")

elapsed = time.monotonic() - t0
fps = good / elapsed if elapsed > 0 else 0

pipeline.stop()

# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'═'*50}")
print(f"  Frames OK : {good}/{NUM_FRAMES}")
print(f"  Elapsed   : {elapsed:.2f}s")
print(f"  FPS       : {fps:.1f}")
print(f"{'═'*50}")

if good == NUM_FRAMES:
    print("✓  RealSense D435i is WORKING on USB 3.0 — all good!")
    sys.exit(0)
elif good > 0:
    print("⚠  Some frames dropped — check USB connection.")
    sys.exit(0)
else:
    print("✗  No frames captured — camera may be faulty.")
    sys.exit(1)
