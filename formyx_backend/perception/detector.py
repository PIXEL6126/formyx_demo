"""
formyx_backend/perception/detector.py
-------------------------------------
Loads the YOLOv8 model and performs real-time object detection
to locate balloon and drone targets in raw video frames.
Supports fast ONNX Runtime inference on Raspberry Pi CPU.

Optimised for **long-range** and **multi-target** scenarios via
MultiScaleDetector (SAHI-style tiled inference with adaptive
confidence thresholds).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import get

try:
    import onnxruntime as ort
except ImportError:
    ort = None

log = logging.getLogger(__name__)


class BalloonDetector:
    """
    YOLOv8-based balloon detector optimized for companion computer (CPU) execution.
    """

    def __init__(self, model_path: str | None = None) -> None:
        cfg_path = model_path or get("perception", "model_path", "models/balloon_detector.pt")
        self.model_path = Path(cfg_path)
        
        self.conf_threshold: float = get("perception", "confidence_threshold", 0.60)
        self.resolution: Tuple[int, int] = tuple(get("perception", "inference_resolution", [320, 320]))
        self.class_id: int = get("perception", "target_class_id", 0)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model weights file not found: {self.model_path.absolute()}\n"
                "Please place the trained balloon_detector.pt file in the models/ directory."
            )
            
        log.info("Loading YOLO model from %s...", self.model_path)
        self.model = YOLO(str(self.model_path))
        log.info("YOLO model loaded successfully on CPU.")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image in OpenCV BGR format.

        Returns
        -------
        List[Dict[str, Any]]
            List of detected balloons. Each detection is a dict with keys:
            * 'bbox': Tuple[float, float, float, float] -> (xmin, ymin, xmax, ymax) in pixels
            * 'center': Tuple[float, float] -> (x_center, y_center) in pixels
            * 'confidence': float -> detection confidence [0.0, 1.0]
            * 'class_id': int -> class index
        """
        if frame is None or frame.size == 0:
            return []

        # Run inference using CPU-optimized parameters
        results = self.model(
            frame,
            imgsz=self.resolution,
            conf=self.conf_threshold,
            device="cpu",
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            # Move data to CPU numpy for quick iteration
            xyxys = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()

            for bbox, conf, cls in zip(xyxys, confs, clss):
                cls_val = int(cls)
                if cls_val != self.class_id:
                    continue

                xmin, ymin, xmax, ymax = map(float, bbox)
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0

                detections.append({
                    "bbox": (xmin, ymin, xmax, ymax),
                    "center": (x_center, y_center),
                    "confidence": float(conf),
                    "class_id": cls_val
                })

        return detections


class ONNXYoloDetector:
    """
    Ultra-fast direct ONNX Runtime detector for YOLO models on CPU.
    Bypasses PyTorch/Ultralytics Python wrapper overhead during inference.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> None:
        if ort is None:
            raise ImportError("onnxruntime package is not installed.")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Optimize Session Options for Raspberry Pi 5 CPU (2 threads avoids scheduling overhead)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape  # [1, 3, H, W]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Runs inference and post-processing, returning predictions in original coordinates."""
        h_orig, w_orig = image.shape[:2]
        
        # 1. Preprocess: Resize, BGR2RGB, transpose HWC -> CHW, and normalize
        input_image = cv2.resize(image, (self.input_w, self.input_h))
        input_data = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_data = input_data.transpose(2, 0, 1)  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0
        
        # 2. Run raw ONNX Runtime inference
        outputs = self.session.run(None, {self.input_name: input_data})
        output = outputs[0]  # shape [1, 4 + C, num_anchors]
        
        # 3. Postprocess: Reshape and extract predictions
        predictions = np.squeeze(output)  # shape [4 + C, num_anchors]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
            
        boxes = predictions[:, :4]  # [cx, cy, w, h]
        scores = predictions[:, 4:]  # [class_scores]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        # Convert box format: [cx, cy, w, h] to [x1, y1, x2, y2]
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Scale back to original resolution
        x_scale = w_orig / self.input_w
        y_scale = h_orig / self.input_h
        
        x1 = (cx - w / 2) * x_scale
        y1 = (cy - h / 2) * y_scale
        x2 = (cx + w / 2) * x_scale
        y2 = (cy + h / 2) * y_scale
        
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        
        # Run Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xyxy.tolist(),
            scores=confidences.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold
        )
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                results.append({
                    "box": list(map(int, boxes_xyxy[idx])),
                    "confidence": float(confidences[idx]),
                    "class_id": int(class_ids[idx])
                })
        return results


class ObjectDetector:
    """
    Dual-class object detector wrapper that detects balloons (class 0) and
    drones (class 1) using either:
      • **OptimizedMultiScaleDetector** — SAHI-style tiled ONNX inference
        with letterbox preprocessing and soft-NMS for high-confidence
        long-range and multi-target detection (default when ONNX models exist)
      • **PyTorch YOLO** — fallback when ONNX is unavailable

    Constructor parameter *enable_tiled* enables/disables the tiled pass
    (set ``False`` on very slow hardware to recover FPS).
    """
    def __init__(self, model_path: str | None = None, enable_tiled: bool = True) -> None:
        cfg_path = model_path or get("perception", "model_path", "models/drone_balloon_detector.pt")
        self.model_path = Path(cfg_path)
        self.target_class_ids = {0, 1}
        self.enable_tiled = enable_tiled

        # Resolve paths to balloon and drone models from config with defaults
        parent_dir = self.model_path.parent
        self.balloon_pt_path = Path(get("perception", "balloon_model_path", str(parent_dir / "yolo11n_balloon.pt")))
        self.drone_pt_path = Path(get("perception", "drone_model_path", str(parent_dir / "yolo11n_drone.pt")))
        
        # ONNX models (preferred for performance)
        self.balloon_onnx_path = self.balloon_pt_path.with_suffix("").parent / f"{self.balloon_pt_path.stem}_320.onnx"
        self.drone_onnx_path = self.drone_pt_path.with_suffix("").parent / f"{self.drone_pt_path.stem}_320.onnx"
        
        # Check if we can use ONNX Runtime
        self.use_onnx = False
        if ort is not None:
            if self.balloon_onnx_path.exists() and self.drone_onnx_path.exists():
                self.use_onnx = True
            
        if self.use_onnx:
            from perception.optimized_detector import OptimizedMultiScaleDetector
            log.info(
                "Initialising OptimizedMultiScaleDetector (letterbox + soft-NMS, tiled=%s) "
                "for high-confidence long-range + multi-target detection.",
                self.enable_tiled,
            )
            self.balloon_detector = OptimizedMultiScaleDetector(
                str(self.balloon_onnx_path),
                conf_threshold=0.18,
                conf_threshold_small=0.10,
                iou_threshold=0.45,
                small_box_area=900,
                enable_tiled=self.enable_tiled,
                tile_size=320,
                tile_overlap=0.25,
            )
            self.drone_detector = OptimizedMultiScaleDetector(
                str(self.drone_onnx_path),
                conf_threshold=0.18,
                conf_threshold_small=0.10,
                iou_threshold=0.45,
                small_box_area=900,
                enable_tiled=self.enable_tiled,
                tile_size=320,
                tile_overlap=0.25,
            )
        else:
            log.info("ONNX models not found or onnxruntime missing. Falling back to PyTorch YOLO.")
            # Check for PyTorch model files with fallback
            if not self.balloon_pt_path.exists():
                alt_path = Path("/home/dart2/Formyxcv/yolo11n_balloon.pt")
                if alt_path.exists():
                    self.balloon_pt_path = alt_path
                else:
                    raise FileNotFoundError(f"Balloon model weights not found at {self.balloon_pt_path}")
            if not self.drone_pt_path.exists():
                alt_path = Path("/home/dart2/Formyxcv/yolo11n_drone.pt")
                if alt_path.exists():
                    self.drone_pt_path = alt_path
                else:
                    raise FileNotFoundError(f"Drone model weights not found at {self.drone_pt_path}")
                    
            self.balloon_model = YOLO(str(self.balloon_pt_path))
            self.drone_model = YOLO(str(self.drone_pt_path))
            
    def detect_balloons(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect balloons (class 0) in the frame."""
        if frame is None or frame.size == 0:
            return []
            
        if self.use_onnx:
            results_list = self.balloon_detector.detect(frame)
            detections = []
            for det in results_list:
                xmin, ymin, xmax, ymax = det["box"]
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                detections.append({
                    "bbox": (float(xmin), float(ymin), float(xmax), float(ymax)),
                    "center": (x_center, y_center),
                    "confidence": float(det["confidence"]),
                    "class_id": 0
                })
            return detections
        else:
            results = self.balloon_model(frame, imgsz=320, conf=0.15, device="cpu", verbose=False)
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                xyxys = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for bbox, conf in zip(xyxys, confs):
                    xmin, ymin, xmax, ymax = map(float, bbox)
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    detections.append({
                        "bbox": (xmin, ymin, xmax, ymax),
                        "center": (x_center, y_center),
                        "confidence": float(conf),
                        "class_id": 0
                    })
            return detections

    def detect_drones(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect drones (class 1) in the frame."""
        if frame is None or frame.size == 0:
            return []
            
        if self.use_onnx:
            results_list = self.drone_detector.detect(frame)
            detections = []
            for det in results_list:
                xmin, ymin, xmax, ymax = det["box"]
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                detections.append({
                    "bbox": (float(xmin), float(ymin), float(xmax), float(ymax)),
                    "center": (x_center, y_center),
                    "confidence": float(det["confidence"]),
                    "class_id": 1  # Target class ID 1 for drone
                })
            return detections
        else:
            results = self.drone_model(frame, imgsz=320, conf=0.15, device="cpu", verbose=False)
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                xyxys = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for bbox, conf in zip(xyxys, confs):
                    xmin, ymin, xmax, ymax = map(float, bbox)
                    x_center = (xmin + xmax) / 2.0
                    y_center = (ymin + ymax) / 2.0
                    detections.append({
                        "bbox": (xmin, ymin, xmax, ymax),
                        "center": (x_center, y_center),
                        "confidence": float(conf),
                        "class_id": 1  # Target class ID 1 for drone
                    })
            return detections

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect balloons AND drones in a single call.

        Runs both models on every invocation and merges results, matching
        the behaviour of the working Formyxcv prototype. The previous
        alternating strategy (one class per frame) caused either balloons or
        drones to be completely invisible on every other frame.
        """
        return self.detect_balloons(frame) + self.detect_drones(frame)

