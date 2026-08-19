"""
tools/evaluate_model.py
-----------------------
Comprehensive evaluation of the balloon & drone ONNX/PT detection models.

Extracts frames from recorded videos in camrec/, runs both the old
single-pass detector and the new multi-scale (tiled) detector, and
produces a full results folder with:

  eval_results/
  ├── summary_report.txt          — text metrics overview
  ├── detection_log.csv           — per-frame detection counts + timings
  ├── fps_comparison.png          — FPS chart: single-pass vs tiled
  ├── detection_counts.png        — detection count comparison chart
  ├── confidence_distribution.png — histogram of detection confidences
  ├── size_distribution.png       — histogram of bounding-box areas
  ├── sample_frames/              — annotated sample frames from each video
  └── latency_percentiles.png     — inference latency percentile chart

Usage:
    cd formyx_backend
    python3 tools/evaluate_model.py [--max-frames 200] [--sample-every 15]
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perception.detector import ONNXYoloDetector
from perception.multi_scale_detector import MultiScaleDetector


# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CAMREC = ROOT / "camrec"
MODELS = ROOT / "models"
BALLOON_ONNX = str(MODELS / "yolo11n_balloon_320.onnx")
DRONE_ONNX   = str(MODELS / "yolo11n_drone_320.onnx")
OUT_DIR = ROOT / "eval_results"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate balloon/drone detection models")
    p.add_argument("--max-frames", type=int, default=200,
                   help="Max frames to sample per video (default 200)")
    p.add_argument("--sample-every", type=int, default=15,
                   help="Sample every Nth frame (default 15)")
    p.add_argument("--output-dir", type=str, default=str(OUT_DIR),
                   help="Output directory for results")
    return p.parse_args()


def collect_videos():
    vids = sorted(CAMREC.glob("*.mp4"))
    # Prefer the larger/real recordings
    return [v for v in vids if v.stat().st_size > 500_000]


def extract_frames(video_path, sample_every, max_frames):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def run_detector(detector, frames, label):
    """Run detector on all frames, return (results_per_frame, latencies_ms)."""
    results = []
    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        results.append(dets)
    return results, latencies


def box_area(det):
    b = det["box"]
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def annotate_frame(frame, dets, label):
    vis = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        conf = d["confidence"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample_frames").mkdir(exist_ok=True)

    videos = collect_videos()
    if not videos:
        print("ERROR: No video files found in camrec/")
        sys.exit(1)

    print(f"Found {len(videos)} videos: {[v.name for v in videos]}")
    print(f"Output directory: {out}")

    # ── Build detectors ──────────────────────────────────────────────
    print("\nLoading detectors...")
    single_balloon = ONNXYoloDetector(BALLOON_ONNX, conf_threshold=0.25)
    single_drone   = ONNXYoloDetector(DRONE_ONNX,   conf_threshold=0.25)

    tiled_balloon = MultiScaleDetector(BALLOON_ONNX, conf_threshold=0.20,
                                       conf_threshold_small=0.12,
                                       enable_tiled=True)
    tiled_drone   = MultiScaleDetector(DRONE_ONNX,   conf_threshold=0.20,
                                       conf_threshold_small=0.12,
                                       enable_tiled=True)

    # ── CSV log ──────────────────────────────────────────────────────
    csv_path = out / "detection_log.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "video", "frame_idx",
        "single_balloon_count", "single_drone_count", "single_latency_ms",
        "tiled_balloon_count",  "tiled_drone_count",  "tiled_latency_ms",
    ])

    # ── Aggregators ──────────────────────────────────────────────────
    all_single_lat = []
    all_tiled_lat  = []
    all_single_counts = []
    all_tiled_counts  = []
    all_single_confs  = []
    all_tiled_confs   = []
    all_single_areas  = []
    all_tiled_areas   = []
    total_frames = 0

    for vi, vpath in enumerate(videos):
        vname = vpath.name
        print(f"\n{'='*60}")
        print(f"Processing {vname} ({vi+1}/{len(videos)})...")
        frames = extract_frames(vpath, args.sample_every, args.max_frames)
        print(f"  Extracted {len(frames)} frames")

        for fi, frame in enumerate(frames):
            total_frames += 1

            # Single-pass
            t0 = time.perf_counter()
            sb = single_balloon.detect(frame)
            sd = single_drone.detect(frame)
            single_ms = (time.perf_counter() - t0) * 1000.0

            # Tiled
            t0 = time.perf_counter()
            tb = tiled_balloon.detect(frame)
            td = tiled_drone.detect(frame)
            tiled_ms = (time.perf_counter() - t0) * 1000.0

            all_single_lat.append(single_ms)
            all_tiled_lat.append(tiled_ms)
            all_single_counts.append(len(sb) + len(sd))
            all_tiled_counts.append(len(tb) + len(td))

            for d in sb + sd:
                all_single_confs.append(d["confidence"])
                all_single_areas.append(box_area(d))
            for d in tb + td:
                all_tiled_confs.append(d["confidence"])
                all_tiled_areas.append(box_area(d))

            writer.writerow([
                vname, fi,
                len(sb), len(sd), f"{single_ms:.1f}",
                len(tb), len(td), f"{tiled_ms:.1f}",
            ])

            # Save sample annotated frames (first 5 per video)
            if fi < 5:
                vis_s = annotate_frame(frame, sb + sd, f"Single-pass | {single_ms:.0f}ms")
                vis_t = annotate_frame(frame, tb + td, f"Tiled/SAHI  | {tiled_ms:.0f}ms")
                combined = np.hstack([vis_s, vis_t])
                sample_path = out / "sample_frames" / f"{vname}_frame{fi:03d}.jpg"
                cv2.imwrite(str(sample_path), combined)

            if (fi + 1) % 50 == 0:
                print(f"  Processed {fi+1}/{len(frames)} frames...")

    csv_file.close()
    print(f"\n{'='*60}")
    print(f"Total frames evaluated: {total_frames}")

    # ── Generate charts ──────────────────────────────────────────────
    print("\nGenerating charts...")
    plt.style.use("dark_background")

    # 1. FPS comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    single_fps = [1000.0 / l for l in all_single_lat]
    tiled_fps  = [1000.0 / l for l in all_tiled_lat]
    ax.plot(single_fps, label=f"Single-pass (avg {np.mean(single_fps):.1f} FPS)", alpha=0.7)
    ax.plot(tiled_fps,  label=f"Tiled/SAHI (avg {np.mean(tiled_fps):.1f} FPS)", alpha=0.7)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("FPS (1000/latency_ms)")
    ax.set_title("Inference FPS: Single-Pass vs Tiled (SAHI)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out / "fps_comparison.png"), dpi=150)
    plt.close(fig)

    # 2. Detection counts comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(all_single_counts, label=f"Single-pass (total {sum(all_single_counts)})", alpha=0.7)
    ax.plot(all_tiled_counts,  label=f"Tiled/SAHI (total {sum(all_tiled_counts)})", alpha=0.7)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Detections per Frame")
    ax.set_title("Detection Counts: Single-Pass vs Tiled")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out / "detection_counts.png"), dpi=150)
    plt.close(fig)

    # 3. Confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if all_single_confs:
        axes[0].hist(all_single_confs, bins=30, color="#FF6B6B", alpha=0.8, edgecolor="white")
    axes[0].set_title("Single-Pass Confidence Distribution")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    if all_tiled_confs:
        axes[1].hist(all_tiled_confs, bins=30, color="#4ECDC4", alpha=0.8, edgecolor="white")
    axes[1].set_title("Tiled/SAHI Confidence Distribution")
    axes[1].set_xlabel("Confidence")
    fig.tight_layout()
    fig.savefig(str(out / "confidence_distribution.png"), dpi=150)
    plt.close(fig)

    # 4. Box size (area) distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if all_single_areas:
        axes[0].hist(all_single_areas, bins=40, color="#FF6B6B", alpha=0.8, edgecolor="white")
    axes[0].set_title("Single-Pass Box Area Distribution")
    axes[0].set_xlabel("Area (px²)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(900, color="yellow", linestyle="--", label="Small threshold (900px²)")
    axes[0].legend()
    if all_tiled_areas:
        axes[1].hist(all_tiled_areas, bins=40, color="#4ECDC4", alpha=0.8, edgecolor="white")
    axes[1].set_title("Tiled/SAHI Box Area Distribution")
    axes[1].set_xlabel("Area (px²)")
    axes[1].axvline(900, color="yellow", linestyle="--", label="Small threshold (900px²)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(str(out / "size_distribution.png"), dpi=150)
    plt.close(fig)

    # 5. Latency percentile chart
    fig, ax = plt.subplots(figsize=(8, 5))
    percs = [50, 75, 90, 95, 99]
    s_percs = np.percentile(all_single_lat, percs)
    t_percs = np.percentile(all_tiled_lat, percs)
    x = np.arange(len(percs))
    w = 0.35
    ax.bar(x - w/2, s_percs, w, label="Single-pass", color="#FF6B6B")
    ax.bar(x + w/2, t_percs, w, label="Tiled/SAHI", color="#4ECDC4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in percs])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency Percentiles")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(out / "latency_percentiles.png"), dpi=150)
    plt.close(fig)

    # ── Summary report ───────────────────────────────────────────────
    small_single = sum(1 for a in all_single_areas if a < 900)
    small_tiled  = sum(1 for a in all_tiled_areas if a < 900)

    report_lines = [
        "=" * 65,
        "  FORMYX BALLOON/DRONE DETECTION — MODEL EVALUATION REPORT",
        "=" * 65,
        "",
        f"  Videos evaluated:       {len(videos)}",
        f"  Total frames sampled:   {total_frames}",
        f"  Models:                 yolo11n_balloon_320.onnx + yolo11n_drone_320.onnx",
        "",
        "─" * 65,
        "  SINGLE-PASS (320×320, old pipeline)",
        "─" * 65,
        f"  Total detections:       {sum(all_single_counts)}",
        f"  Avg detections/frame:   {np.mean(all_single_counts):.2f}",
        f"  Small detections (<900px²): {small_single}",
        f"  Avg confidence:         {np.mean(all_single_confs):.3f}" if all_single_confs else "  Avg confidence:         N/A",
        f"  Avg latency:            {np.mean(all_single_lat):.1f} ms",
        f"  Avg FPS:                {1000.0 / np.mean(all_single_lat):.1f}",
        f"  P50 / P95 / P99 lat:   {np.percentile(all_single_lat, 50):.1f} / {np.percentile(all_single_lat, 95):.1f} / {np.percentile(all_single_lat, 99):.1f} ms",
        "",
        "─" * 65,
        "  TILED / SAHI (multi-scale, new pipeline)",
        "─" * 65,
        f"  Total detections:       {sum(all_tiled_counts)}",
        f"  Avg detections/frame:   {np.mean(all_tiled_counts):.2f}",
        f"  Small detections (<900px²): {small_tiled}",
        f"  Avg confidence:         {np.mean(all_tiled_confs):.3f}" if all_tiled_confs else "  Avg confidence:         N/A",
        f"  Avg latency:            {np.mean(all_tiled_lat):.1f} ms",
        f"  Avg FPS:                {1000.0 / np.mean(all_tiled_lat):.1f}",
        f"  P50 / P95 / P99 lat:   {np.percentile(all_tiled_lat, 50):.1f} / {np.percentile(all_tiled_lat, 95):.1f} / {np.percentile(all_tiled_lat, 99):.1f} ms",
        "",
        "─" * 65,
        "  IMPROVEMENT",
        "─" * 65,
        f"  Detection count gain:   {sum(all_tiled_counts) - sum(all_single_counts):+d} ({((sum(all_tiled_counts) / max(sum(all_single_counts),1)) - 1) * 100:+.1f}%)",
        f"  Small-target gain:      {small_tiled - small_single:+d} more small detections",
        f"  FPS cost:               {1000.0/np.mean(all_tiled_lat) - 1000.0/np.mean(all_single_lat):+.1f} FPS",
        "",
        "=" * 65,
        "  Output files:",
        f"    {out / 'detection_log.csv'}",
        f"    {out / 'fps_comparison.png'}",
        f"    {out / 'detection_counts.png'}",
        f"    {out / 'confidence_distribution.png'}",
        f"    {out / 'size_distribution.png'}",
        f"    {out / 'latency_percentiles.png'}",
        f"    {out / 'sample_frames/'}  ({len(list((out/'sample_frames').glob('*.jpg')))} images)",
        "=" * 65,
    ]

    report = "\n".join(report_lines)
    print("\n" + report)

    with open(out / "summary_report.txt", "w") as f:
        f.write(report + "\n")

    print(f"\n✓ All results saved to: {out}/")


if __name__ == "__main__":
    main()
