#!/usr/bin/env python3
"""
tools/train_balloon_optimized.py
---------------------------------
Optimized balloon detection training pipeline.

Key improvements over original train_balloon.py:
  1. Trains for 150 epochs (was 30) — small datasets need more epochs
  2. Uses yolo11s (small) instead of yolo11n (nano) for better accuracy
  3. Heavy augmentation for small-dataset generalization
  4. Lower initial learning rate (0.001) to avoid overshooting
  5. Cosine LR schedule for smoother convergence
  6. Freeze backbone for first 10 epochs (transfer learning best practice)
  7. Mixed precision (AMP) for faster training
  8. Proper patience for early stopping
  9. Fixes data.yaml path to use Linux absolute path
 10. Exports optimized ONNX models at multiple resolutions

Usage:
    python tools/train_balloon_optimized.py [--model yolo11s.pt] [--epochs 150] [--imgsz 640]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def fix_data_yaml(dataset_dir: str) -> str:
    """Ensure data.yaml has the correct absolute Linux path."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    abs_dataset = os.path.abspath(dataset_dir)

    yaml_content = f"""path: {abs_dataset}
train: train/images
val: val/images

nc: 1
names:
  0: balloon
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[+] Fixed data.yaml path → {abs_dataset}")
    return yaml_path


def augment_dataset_offline(dataset_dir: str) -> None:
    """
    Create additional training samples via offline augmentation.
    This multiplies the effective dataset size ~3x using:
      - horizontal flip
      - brightness/contrast jitter
      - gaussian blur
    Labels are adjusted accordingly for flips.
    """
    import cv2
    import numpy as np

    images_dir = os.path.join(dataset_dir, "train", "images")
    labels_dir = os.path.join(dataset_dir, "train", "labels")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    orig_count = len(image_files)
    augmented = 0

    print(f"[*] Starting offline augmentation on {orig_count} training images...")

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        with open(label_path, "r") as f:
            labels = f.readlines()

        base_name = os.path.splitext(img_file)[0]

        # --- Augmentation 1: Horizontal Flip ---
        flipped = cv2.flip(img, 1)
        flip_labels = []
        for line in labels:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = parts[0]
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                # Flip x coordinate
                cx_flipped = 1.0 - cx
                flip_labels.append(f"{cls_id} {cx_flipped:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        aug_img_path = os.path.join(images_dir, f"{base_name}_hflip.jpg")
        aug_lbl_path = os.path.join(labels_dir, f"{base_name}_hflip.txt")
        if not os.path.exists(aug_img_path):
            cv2.imwrite(aug_img_path, flipped)
            with open(aug_lbl_path, "w") as f:
                f.writelines(flip_labels)
            augmented += 1

        # --- Augmentation 2: Brightness + Contrast jitter ---
        alpha = np.random.uniform(0.7, 1.3)  # contrast
        beta = np.random.randint(-30, 30)     # brightness
        jittered = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        aug_img_path = os.path.join(images_dir, f"{base_name}_jitter.jpg")
        aug_lbl_path = os.path.join(labels_dir, f"{base_name}_jitter.txt")
        if not os.path.exists(aug_img_path):
            cv2.imwrite(aug_img_path, jittered)
            with open(aug_lbl_path, "w") as f:
                f.writelines(labels)  # Same labels since no geometric transform
            augmented += 1

        # --- Augmentation 3: Slight Gaussian blur (simulates distance/motion) ---
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        # Also add slight random crop/zoom
        h_img, w_img = blurred.shape[:2]
        crop_pct = np.random.uniform(0.85, 0.95)
        new_h = int(h_img * crop_pct)
        new_w = int(w_img * crop_pct)
        y_off = np.random.randint(0, h_img - new_h + 1)
        x_off = np.random.randint(0, w_img - new_w + 1)
        cropped = blurred[y_off:y_off + new_h, x_off:x_off + new_w]
        cropped = cv2.resize(cropped, (w_img, h_img))

        # Adjust labels for the crop
        crop_labels = []
        for line in labels:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = parts[0]
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # Transform center to crop coordinates
                cx_new = (cx * w_img - x_off) / new_w
                cy_new = (cy * h_img - y_off) / new_h
                w_new = w * w_img / new_w
                h_new = h * h_img / new_h

                # Clamp to [0,1] and skip if center is outside
                if 0 < cx_new < 1 and 0 < cy_new < 1:
                    w_new = min(w_new, min(cx_new, 1 - cx_new) * 2)
                    h_new = min(h_new, min(cy_new, 1 - cy_new) * 2)
                    if w_new > 0.01 and h_new > 0.01:
                        crop_labels.append(
                            f"{cls_id} {cx_new:.6f} {cy_new:.6f} {w_new:.6f} {h_new:.6f}\n"
                        )

        if crop_labels:
            aug_img_path = os.path.join(images_dir, f"{base_name}_crop.jpg")
            aug_lbl_path = os.path.join(labels_dir, f"{base_name}_crop.txt")
            if not os.path.exists(aug_img_path):
                cv2.imwrite(aug_img_path, cropped)
                with open(aug_lbl_path, "w") as f:
                    f.writelines(crop_labels)
                augmented += 1

    new_total = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"[+] Offline augmentation complete: {orig_count} → {new_total} training images (+{augmented} new)")


def train(args):
    """Run the optimized training pipeline."""
    from ultralytics import YOLO

    dataset_dir = args.dataset_dir

    if not os.path.isdir(dataset_dir):
        print(f"[-] Error: Dataset directory not found: {dataset_dir}")
        print("[-] Please run prepare_dataset.py first.")
        return

    # Step 1: Fix data.yaml
    yaml_path = fix_data_yaml(dataset_dir)

    # Step 2: Offline augmentation (only on first run — skips existing augmented files)
    if not args.skip_augment:
        augment_dataset_offline(dataset_dir)

    # Step 3: Remove stale label cache (forces re-scan after augmentation)
    for split in ["train", "val"]:
        cache_file = os.path.join(dataset_dir, split, "labels.cache")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"[+] Removed stale cache: {cache_file}")

    # Step 4: Load model
    print(f"\n[*] Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    # Step 5: Train with optimized hyperparameters
    print(f"\n{'='*70}")
    print(f"  OPTIMIZED BALLOON TRAINING")
    print(f"  Model:      {args.model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Device:     {'cpu' if not args.gpu else 'auto'}")
    print(f"  LR:         {args.lr}")
    print(f"{'='*70}\n")

    device = "cpu" if not args.gpu else "0"

    try:
        results = model.train(
            data=yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=2,           # Lower workers on Pi to avoid memory pressure
            project="runs",
            name="balloon_optimized",
            exist_ok=True,

            # ── Learning Rate ──
            lr0=args.lr,         # Lower initial LR for small dataset fine-tuning
            lrf=0.01,            # Final LR = lr0 * lrf (cosine decay)
            cos_lr=True,         # Cosine LR schedule (smoother than step)

            # ── Regularization ──
            weight_decay=0.001,  # Slightly higher than default (0.0005) for small dataset
            dropout=0.1,         # Light dropout for regularization

            # ── Freeze backbone for transfer learning ──
            freeze=10 if args.freeze_backbone else None,  # Freeze first 10 layers

            # ── Augmentation (aggressive for small dataset) ──
            mosaic=1.0,          # Mosaic augmentation (combines 4 images)
            mixup=0.15,          # MixUp augmentation (blends 2 images)
            copy_paste=0.1,      # Copy-paste augmentation
            degrees=15.0,        # Random rotation ±15°
            translate=0.2,       # Random translation ±20%
            scale=0.7,           # Random scale ±70%
            shear=5.0,           # Random shear ±5°
            perspective=0.001,   # Slight perspective distortion
            flipud=0.1,          # Vertical flip (balloons can be upside-down)
            fliplr=0.5,          # Horizontal flip
            hsv_h=0.02,          # Hue augmentation
            hsv_s=0.8,           # Saturation augmentation
            hsv_v=0.5,           # Value (brightness) augmentation
            erasing=0.3,         # Random erasing

            # ── Early stopping ──
            patience=40,         # Stop if no improvement for 40 epochs

            # ── Close mosaic early ──
            close_mosaic=20,     # Disable mosaic for last 20 epochs (fine-tune)

            # ── Loss weights ──
            box=7.5,             # Box regression loss weight
            cls=1.0,             # Classification loss weight (higher for single class)
            dfl=1.5,             # Distribution focal loss weight

            # ── Misc ──
            amp=True,            # Mixed precision (faster on ARM)
            cache=True if args.cache else False,  # Cache images in RAM
            rect=False,          # Rectangular training (disabled for mosaic)
            single_cls=True,     # Single class mode (balloon only)
            verbose=True,
            plots=True,
        )

        print("\n[+] Training completed successfully!")

        # Step 6: Copy best weights
        best_src = "runs/balloon_optimized/weights/best.pt"
        last_src = "runs/balloon_optimized/weights/last.pt"
        best_dst = args.output_weights

        if os.path.exists(best_src):
            shutil.copy(best_src, best_dst)
            print(f"[+] Best weights → {best_dst}")
        elif os.path.exists(last_src):
            shutil.copy(last_src, best_dst)
            print(f"[+] Last weights → {best_dst} (best not found, used last)")

        # Step 7: Export ONNX models
        if not args.skip_export:
            export_onnx(best_dst, args.imgsz)

    except Exception as e:
        print(f"[-] Error during training: {e}")
        import traceback
        traceback.print_exc()


def export_onnx(weights_path: str, train_imgsz: int = 640):
    """Export trained .pt model to ONNX at multiple resolutions."""
    from ultralytics import YOLO

    if not os.path.exists(weights_path):
        print(f"[-] Weights not found at {weights_path}, skipping ONNX export.")
        return

    model = YOLO(weights_path)
    base_name = Path(weights_path).stem

    # Export at inference resolution (320) and training resolution (640)
    for sz in [320, 640]:
        print(f"\n[*] Exporting ONNX at {sz}x{sz}...")
        try:
            exported = model.export(
                format="onnx",
                imgsz=sz,
                simplify=True,
                opset=13,
                half=False,
                dynamic=False,
            )
            if exported:
                # Move to standard location
                onnx_src = exported
                onnx_dst = f"{base_name}_{sz}.onnx"
                if os.path.exists(str(onnx_src)):
                    shutil.copy(str(onnx_src), onnx_dst)
                    print(f"[+] ONNX exported → {onnx_dst}")

                    # Also copy to models/ directory in formyx_backend
                    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                    if os.path.isdir(models_dir):
                        shutil.copy(onnx_dst, os.path.join(models_dir, f"yolo11n_balloon_{sz}.onnx"))
                        print(f"[+] Also copied to models/{base_name}_{sz}.onnx")

        except Exception as e:
            print(f"[-] ONNX export at {sz} failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Balloon Detection Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training on Raspberry Pi (CPU):
  python tools/train_balloon_optimized.py --epochs 80 --batch 8

  # Full training on desktop with GPU:
  python tools/train_balloon_optimized.py --gpu --epochs 200 --batch 32 --imgsz 640

  # Use larger model for better accuracy:
  python tools/train_balloon_optimized.py --model yolo11s.pt --epochs 150

  # Skip augmentation (if already done):
  python tools/train_balloon_optimized.py --skip-augment --epochs 100
        """
    )
    parser.add_argument("--model", default="yolo11s.pt",
                        help="Base model to fine-tune (default: yolo11s.pt for better accuracy)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs (default: 150)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8 for Pi, use 16-32 on GPU)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training (default: CPU)")
    parser.add_argument("--cache", action="store_true",
                        help="Cache training images in RAM (faster but uses more memory)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze backbone layers for transfer learning (default: True)")
    parser.add_argument("--no-freeze", action="store_false", dest="freeze_backbone",
                        help="Don't freeze backbone layers")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Skip offline data augmentation step")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip ONNX export after training")
    parser.add_argument("--dataset-dir", default=None,
                        help="Path to balloon_dataset directory")
    parser.add_argument("--output-weights", default="yolo11s_balloon_optimized.pt",
                        help="Output path for best weights (default: yolo11s_balloon_optimized.pt)")

    args = parser.parse_args()

    # Auto-detect dataset directory
    if args.dataset_dir is None:
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "balloon_dataset"),
            "/home/dart2/Formyxcv/balloon_dataset",
            "balloon_dataset",
        ]
        for c in candidates:
            if os.path.isdir(c):
                args.dataset_dir = c
                break
        if args.dataset_dir is None:
            print("[-] Could not find balloon_dataset directory. Use --dataset-dir to specify.")
            sys.exit(1)

    print(f"[*] Dataset directory: {args.dataset_dir}")
    train(args)


if __name__ == "__main__":
    main()
