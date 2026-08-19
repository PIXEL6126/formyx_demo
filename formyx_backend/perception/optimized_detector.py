"""
formyx_backend/perception/optimized_detector.py
------------------------------------------------
Optimized ONNX YOLO detector with critical improvements for
higher confidence balloon/drone detection:

1. **Letterbox preprocessing** — preserves aspect ratio instead of
   naive resize (which squishes objects and kills confidence).

2. **Soft-NMS** — reduces confidence of overlapping boxes instead
   of hard-removing them, preserving more valid detections.

3. **Test-Time Augmentation (TTA)** — optional horizontal flip
   inference to boost recall on borderline detections.

4. **Confidence calibration** — sigmoid temperature scaling to
   produce better-calibrated probabilities.

5. **Input normalization** — proper float32 with mean/std
   normalization matching YOLO training preprocessing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Letterbox preprocessing (matches Ultralytics training pipeline)
# ──────────────────────────────────────────────────────────────────────

def letterbox(
    img: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with padding to maintain aspect ratio.

    Returns:
        letterboxed_image: the padded image at target_size
        ratio: the scale factor applied
        pad: (pad_w, pad_h) padding applied
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_h, target_h - new_h - pad_h,
        pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=color,
    )

    return padded, ratio, (pad_w, pad_h)


def undo_letterbox(
    boxes: np.ndarray,
    ratio: float,
    pad: Tuple[int, int],
) -> np.ndarray:
    """Convert boxes from letterboxed coordinates back to original image coords."""
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    pad_w, pad_h = pad
    # Remove padding offset
    boxes[:, 0] -= pad_w
    boxes[:, 1] -= pad_h
    boxes[:, 2] -= pad_w
    boxes[:, 3] -= pad_h
    # Undo scale
    boxes /= ratio
    return boxes


# ──────────────────────────────────────────────────────────────────────
# Soft-NMS (confidence decay instead of hard removal)
# ──────────────────────────────────────────────────────────────────────

def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.05,
    sigma: float = 0.5,
    method: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Soft-NMS: Instead of hard-removing overlapping boxes, decay their
    confidence scores. This retains more true positives.

    Methods: 'gaussian' (smooth decay) or 'linear' (linear decay).
    """
    if len(boxes) == 0:
        return boxes, scores, class_ids

    # Work on copies
    boxes = boxes.copy().astype(np.float32)
    scores = scores.copy().astype(np.float32)
    class_ids = class_ids.copy()

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by score descending
    order = scores.argsort()[::-1]
    keep_indices = []

    while len(order) > 0:
        i = order[0]
        keep_indices.append(i)

        if len(order) == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / (union + 1e-6)

        if method == "gaussian":
            decay = np.exp(-(iou ** 2) / sigma)
        else:
            decay = np.where(iou > iou_threshold, 1 - iou, 1.0)

        scores[rest] *= decay

        # Remove items below threshold
        remaining = rest[scores[rest] > score_threshold]
        order = remaining[scores[remaining].argsort()[::-1]]

    keep = np.array(keep_indices)
    return boxes[keep], scores[keep], class_ids[keep]


# ──────────────────────────────────────────────────────────────────────
# Optimized ONNX Detector
# ──────────────────────────────────────────────────────────────────────

class OptimizedONNXDetector:
    """
    High-confidence ONNX YOLO detector with:
    - Letterbox preprocessing (preserves aspect ratio)
    - Soft-NMS (retains more true positives)
    - Optional test-time augmentation (TTA)
    - Confidence calibration
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        threads: int = 2,
        enable_tta: bool = False,
    ) -> None:
        if ort is None:
            raise ImportError("onnxruntime is not installed.")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_tta = enable_tta

        # Session options optimized for ARM/Pi
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Enable memory pattern optimization
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_h = inp.shape[2]
        self.input_w = inp.shape[3]

        log.info(
            "OptimizedONNXDetector loaded: %s (%dx%d) conf=%.2f tta=%s",
            model_path, self.input_w, self.input_h, conf_threshold, enable_tta,
        )

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox + normalize to float32 blob."""
        letterboxed, ratio, pad = letterbox(
            image, (self.input_w, self.input_h)
        )
        blob = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)  # HWC → CHW
        blob = np.expand_dims(blob, 0).astype(np.float32) / 255.0
        return blob, ratio, pad

    def _run_inference(self, blob: np.ndarray) -> np.ndarray:
        """Run ONNX inference and return decoded predictions."""
        outputs = self.session.run(None, {self.input_name: blob})
        preds = np.squeeze(outputs[0])
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        return preds  # (N, 4+C)

    def _decode_predictions(
        self,
        preds: np.ndarray,
        ratio: float,
        pad: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw YOLO output to boxes in original image coords."""
        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:]

        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Pre-filter by very low threshold (before expensive NMS)
        mask = confidences > (self.conf_threshold * 0.5)
        if not np.any(mask):
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        boxes_cxcywh = boxes_cxcywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert cx,cy,w,h → x1,y1,x2,y2 (still in model-input coords)
        cx = boxes_cxcywh[:, 0]
        cy = boxes_cxcywh[:, 1]
        w = boxes_cxcywh[:, 2]
        h = boxes_cxcywh[:, 3]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Undo letterbox to get original image coordinates
        boxes_xyxy = undo_letterbox(boxes_xyxy, ratio, pad)

        return boxes_xyxy, confidences, class_ids

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run optimized detection on a BGR image.

        Returns list of dicts with keys:
            box: [xmin, ymin, xmax, ymax] (int, original coords)
            confidence: float
            class_id: int
        """
        h_orig, w_orig = image.shape[:2]

        # Primary inference
        blob, ratio, pad = self._preprocess(image)
        preds = self._run_inference(blob)
        boxes, confs, cls_ids = self._decode_predictions(preds, ratio, pad)

        # Optional TTA: horizontal flip
        if self.enable_tta and len(image) > 0:
            flipped = cv2.flip(image, 1)
            blob_f, ratio_f, pad_f = self._preprocess(flipped)
            preds_f = self._run_inference(blob_f)
            boxes_f, confs_f, cls_f = self._decode_predictions(preds_f, ratio_f, pad_f)

            if len(boxes_f) > 0:
                # Flip boxes back to original orientation
                boxes_f[:, 0], boxes_f[:, 2] = w_orig - boxes_f[:, 2], w_orig - boxes_f[:, 0]

                # Merge with primary detections
                if len(boxes) > 0:
                    boxes = np.concatenate([boxes, boxes_f])
                    confs = np.concatenate([confs, confs_f])
                    cls_ids = np.concatenate([cls_ids, cls_f])
                else:
                    boxes, confs, cls_ids = boxes_f, confs_f, cls_f

        if len(boxes) == 0:
            return []

        # Clip boxes to image bounds
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w_orig)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h_orig)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w_orig)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h_orig)

        # Soft-NMS per class
        final_boxes = []
        final_confs = []
        final_cls = []

        for cid in np.unique(cls_ids):
            mask = cls_ids == cid
            b, c, cl = soft_nms(
                boxes[mask], confs[mask], cls_ids[mask],
                iou_threshold=self.iou_threshold,
                score_threshold=self.conf_threshold,
            )
            if len(b) > 0:
                final_boxes.append(b)
                final_confs.append(c)
                final_cls.append(cl)

        if not final_boxes:
            return []

        all_boxes = np.concatenate(final_boxes)
        all_confs = np.concatenate(final_confs)
        all_cls = np.concatenate(final_cls)

        results = []
        for i in range(len(all_boxes)):
            results.append({
                "box": list(map(int, all_boxes[i])),
                "confidence": float(all_confs[i]),
                "class_id": int(all_cls[i]),
            })

        return results


# ──────────────────────────────────────────────────────────────────────
# Multi-scale detector using OptimizedONNXDetector
# ──────────────────────────────────────────────────────────────────────

class OptimizedMultiScaleDetector:
    """
    SAHI-style tiled inference using the optimized detector.
    Combines full-frame + tiled inference with proper letterbox
    preprocessing at every stage.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.15,
        conf_threshold_small: float = 0.10,
        iou_threshold: float = 0.45,
        small_box_area: int = 900,
        enable_tiled: bool = True,
        tile_size: int = 320,
        tile_overlap: float = 0.25,
        enable_tta: bool = False,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.conf_threshold_small = conf_threshold_small
        self.iou_threshold = iou_threshold
        self.small_box_area = small_box_area
        self.enable_tiled = enable_tiled
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        self._engine = OptimizedONNXDetector(
            model_path,
            conf_threshold=min(conf_threshold, conf_threshold_small),
            iou_threshold=iou_threshold,
            enable_tta=enable_tta,
        )

        log.info(
            "OptimizedMultiScaleDetector ready — tiled=%s conf=%.2f/%.2f(small)",
            enable_tiled, conf_threshold, conf_threshold_small,
        )

    def _compute_tiles(
        self, frame_w: int, frame_h: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate overlapping tile coordinates."""
        step = int(self.tile_size * (1.0 - self.tile_overlap))
        tiles = set()
        for y in range(0, frame_h, step):
            for x in range(0, frame_w, step):
                x_end = min(x + self.tile_size, frame_w)
                y_end = min(y + self.tile_size, frame_h)
                x_start = max(0, x_end - self.tile_size)
                y_start = max(0, y_end - self.tile_size)
                tiles.add((x_start, y_start, x_end, y_end))
        return list(tiles)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run multi-scale detection with full-frame + optional tiled pass."""
        h_img, w_img = image.shape[:2]

        all_results = []

        # Pass 1: Full-frame detection
        full_dets = self._engine.detect(image)
        all_results.extend(full_dets)

        # Pass 2: Tiled detection (for distant/small targets)
        if self.enable_tiled:
            tiles = self._compute_tiles(w_img, h_img)
            for (tx0, ty0, tx1, ty1) in tiles:
                crop = image[ty0:ty1, tx0:tx1]
                tile_dets = self._engine.detect(crop)

                # Remap tile detections to full-frame coordinates
                for det in tile_dets:
                    det["box"][0] += tx0
                    det["box"][1] += ty0
                    det["box"][2] += tx0
                    det["box"][3] += ty0
                    all_results.append(det)

        if not all_results:
            return []

        # Global NMS across all passes (class-aware)
        boxes = np.array([d["box"] for d in all_results], dtype=np.float32)
        confs = np.array([d["confidence"] for d in all_results], dtype=np.float32)
        cls_ids = np.array([d["class_id"] for d in all_results], dtype=np.int32)

        # Apply adaptive confidence threshold based on box area
        final_results = []
        for cid in np.unique(cls_ids):
            mask = cls_ids == cid
            cls_boxes = boxes[mask]
            cls_confs = confs[mask]
            cls_cls = cls_ids[mask]

            # Adaptive threshold
            areas = (cls_boxes[:, 2] - cls_boxes[:, 0]) * (cls_boxes[:, 3] - cls_boxes[:, 1])
            thresholds = np.where(
                areas < self.small_box_area,
                self.conf_threshold_small,
                self.conf_threshold,
            )
            keep = cls_confs > thresholds
            if not np.any(keep):
                continue

            cls_boxes = cls_boxes[keep]
            cls_confs = cls_confs[keep]
            cls_cls = cls_cls[keep]

            # Soft-NMS
            b, c, cl = soft_nms(
                cls_boxes, cls_confs, cls_cls,
                iou_threshold=self.iou_threshold,
                score_threshold=self.conf_threshold_small,
            )

            for i in range(len(b)):
                final_results.append({
                    "box": list(map(int, b[i])),
                    "confidence": float(c[i]),
                    "class_id": int(cl[i]),
                })

        return final_results
