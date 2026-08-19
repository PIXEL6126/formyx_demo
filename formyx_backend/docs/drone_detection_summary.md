# Drone Detection CV Pipeline Summary

This document provides a technical summary of the computer vision pipeline used for drone and balloon (proxy) detection and 3D spatial tracking.

---

## 1. Dataset Details

### Custom Balloon Dataset (Proxy Tracking)
*   **Total Images**: 74 images (annotated in [balloon-data.csv](file:///d:/formyx%20drone/archive/balloon-data.csv)).
*   **Classes**: 1 class (`balloon`). Used as a flight target proxy for local 3D tracking testing.
*   **Split**: 80% Train (59 images) / 20% Validation (15 images) / 0% Test (manually split via [prepare_dataset.py](file:///d:/formyx%20drone/prepare_dataset.py)).
*   **Type**: Real-world images sourced from Roboflow and Open Images Dataset v6 (OIDv6).

### Drone Detection Dataset
*   **Total Images**: ~56,821 images (source: Maciej Pawełczyk and Marek Wojtyra, 2020).
*   **Classes**: 1 class (`drone`).
*   **Split**: ~80% Train / 20% Validation.
*   **Type**: Real-world frames extracted from public YouTube videos under diverse conditions (outdoor flight, cluttered backgrounds).

---

## 2. Model Details

*   **Architecture**: Ultralytics **YOLOv11 nano** (`YOLO11n`) for maximum throughput on edge hardware.
    *   **Drone Model Weights**: `yolo11n_drone.pt` (marie-kjelberg/drone-detector)
    *   **Balloon Model Weights**: `yolo11n_balloon.pt` (locally fine-tuned)
*   **Image Size (Training)**: $640 \times 640$ pixels.
*   **Batch Size**: 16 (for local balloon fine-tuning).
*   **Epochs**:
    *   **Drone Model**: 150 epochs.
    *   **Balloon Model**: 30 epochs (trained via [train_balloon.py](file:///d:/formyx%20drone/train_balloon.py)).
*   **Augmentations**: Standard YOLOv11 augmentations (Mosaic, MixUp, HSV jitter, random scale/translate, flipping) to combat overfitting on small dataset sizes.

---

## 3. Training Metrics (Balloon Fine-Tuning)
Metrics taken at final Epoch 30 (see [results.csv](file:///d:/formyx%20drone/runs/detect/runs/yolo11n_balloon/results.csv)):

| Metric | Value | percentage |
| :--- | :--- | :--- |
| **Precision** | `1.0000` | 100.0% |
| **Recall** | `0.8957` | 89.6% |
| **F1 Score** | `0.9449` | 94.5% |
| **mAP@0.5** | `0.9756` | 97.6% |
| **mAP@0.5:0.95** | `0.8818` | 88.2% |

---

## 4. Inference & Tracking Pipeline

### Inference Configuration
*   **Input Resolution**: Default $320 \times 320$ pixels inside the real-time tracking loops (to maintain high FPS on edge CPU), with option for $640 \times 640$ for accuracy.
*   **Confidence Threshold**: 0.25 (default).
*   **NMS IoU Threshold**: 0.70 (standard YOLOv11 NMS).
*   **SAHI (Slicing Aided Hyper Inference)**: Not used.

### 3D Spatial Tracking
A custom [Tracker3D](file:///d:/formyx%20drone/main.py#L230-L318) manages active targets via a 3D Constant Velocity Kalman Filter ([KalmanFilter3D](file:///d:/formyx%20drone/main.py#L76-L229)):
1.  **Deprojection**: Pixel coordinates of target bounding box centers are aligned with the Intel RealSense D435i depth frame and deprojected to relative 3D spatial coordinates $(x, y, z)$ in meters using camera intrinsics.
2.  **State Representation**: $S = [p_x, p_y, p_z, v_x, v_y, v_z]^T$ (smoothed 3D position and velocity).
3.  **Data Association**: Mahalanobis distance gating (gating threshold defaults to 3.5 standard deviations) to associate incoming YOLO detections with active Kalman filters.
4.  **Ego-Motion Compensation**: Compensates for camera rotational/translational motion when integrated on the companion computer of an active flight controller.

### Performance on Raspberry Pi 5 CPU
*   **Single Model (320px resolution)**: ~15-25 FPS.
*   **Single Model (640px resolution)**: ~5-10 FPS.
*   **Dual Model (Drone + Balloon Sequential Inference)**: ~7-12 FPS at 320px.

---

## 5. Current Limitations

> [!WARNING]
> **Depth Sensor Noise Growth**
> The RealSense D435i's depth measurement noise grows quadratically with range:
> $$\sigma_z = 0.01 + 0.015 \cdot z^2$$
> Targets beyond 10 meters return invalid depth (0), forcing the tracking filter to rely purely on 2D prediction.

*   **Small Target Downscaling**: Downscaling to $320 \times 320$ resolution for real-time edge processing causes small/distant targets to lose features, resulting in miss-detections.
*   **Sensor Shadows & Occlusions**: Target occlusions longer than 15 frames result in track pruning. Additionally, depth shadows on the object boundary can cause noisy distance estimation.
*   **CPU Bottleneck**: Sequential processing of dual models on the Pi 5 CPU limits the control loop update frequency for autonomous guiding commands.

---

## 6. Planned Improvements

*   **Resolution Increase**: Optimize model inference pipeline to run at $640 \times 640$ resolution to improve long-range sensitivity.
*   **SAHI Integration**: Integrate Slicing Aided Hyper Inference (SAHI) to patch large images, improving detection of far-off drones.
*   **Edge Accelerator Support**: Compile the model weights for NPUs (e.g., the Raspberry Pi AI Kit / Hailo-8L chip) to run full $640 \times 640$ dual-model tracking at >30 FPS.
*   **Training Expansion**: Incorporate larger multi-class aerial datasets (like QuincySorrentino's AeroYOLO) to minimize false positives from birds and background clutter.
