# Formyx Backend

**Autonomous balloon-tracking drone** backend for Raspberry Pi 5 with Intel RealSense D435i and Radiolink PIX6 flight controller.

## Architecture

```
formyx_backend/
├── main.py               — Full pipeline: MAVLink + perception + tracking
├── camcv.py              — Standalone camera viewer + recorder
├── logs.py               — Passive MAVLink flight-log recorder
│
├── config/               — YAML-based configuration system
├── depth/                — Intel RealSense D435i depth camera interface
├── perception/           — YOLO object detection (balloon + drone)
├── tracking/             — 3D Kalman multi-target tracker
├── mavlink_interface/    — MAVLink connection + commands (PIX6)
├── navigation/           — Follow controller + search patterns
├── safety/               — Battery/GPS failsafe monitor
├── mission_manager/      — State machine for autonomous missions
├── logging_system/       — BlackBox flight data logger
│
├── models/               — Trained YOLO weights (.pt + .onnx)
├── tools/                — Development utilities (training, testing, streaming)
├── training/             — Dataset preparation + training scripts
├── services/             — systemd service files for auto-start
├── tests/                — Unit tests + hardware test plans
└── docs/                 — Project documentation
```

## Quick Start

### Camera + Detection (no flight controller):
```bash
python main.py --no-mavlink
```

### Full System (with PIX6):
```bash
python main.py --connection serial:/dev/ttyAMA0:921600
```

### Camera Viewer Only:
```bash
python camcv.py
```

### Flight Log Recorder:
```bash
python logs.py
```

## Detection Models

| Model | Class | Format | Size |
|-------|-------|--------|------|
| `yolo11n_balloon.pt` | Balloon | PyTorch | ~5.5MB |
| `yolo11n_balloon_320.onnx` | Balloon | ONNX (320×320) | ~10MB |
| `yolo11n_drone.pt` | Drone | PyTorch | ~5.5MB |
| `yolo11n_drone_320.onnx` | Drone | ONNX (320×320) | ~10MB |

### Retraining
```bash
python tools/train_balloon_optimized.py --epochs 80 --batch 4
```

## Hardware

- **Companion Computer:** Raspberry Pi 5 (8GB)
- **Depth Camera:** Intel RealSense D435i (USB 3.0)
- **Flight Controller:** Radiolink PIX6 (ArduPilot)
- **AI Accelerator:** Hailo-8L AI HAT (optional)

## Configuration

Edit `config/settings.yaml` to tune:
- MAVLink connection parameters
- Detection confidence thresholds
- Tracking filter parameters
- Safety geofence limits
- Navigation speeds
