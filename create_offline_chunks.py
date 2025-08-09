"""
CLI to create offline per-chunk PI3 results and save them to disk.

Example:
  python create_offline_chunks.py \
    --images /path/to/images \
    --model-path Ruicheng/pi3-dinov2-base \
    --output /tmp/run_pi3_chunks \
    --chunk-length 80 --overlap 10 \
    --device cuda --metric-depth --keypoints aliked --max-kp 512
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

from slam.offline_chunk_creator import OfflineChunkCreator, OfflineCreatorConfig


def list_images(root: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files: List[str] = []
    if os.path.isdir(root):
        for ext in exts:
            files.extend(sorted(glob.glob(os.path.join(root, ext))))
    elif os.path.isfile(root):
        # Treat as a text file with paths
        with open(root, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        # Glob pattern
        files = sorted(glob.glob(root))
    return files


def main():
    parser = argparse.ArgumentParser(description="Create offline PI3 chunks and save to disk")
    parser.add_argument("--images", default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist/", help="Folder with images, a glob pattern, or a text file listing image paths")
    parser.add_argument("--model-path", default="yyfz233/Pi3", help="Pi3 model identifier or local path for Pi3.from_pretrained")
    parser.add_argument("--output", default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist/chunks", help="Output directory")
    parser.add_argument("--chunk-length", type=int, default=50)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-depth", action="store_true", help="Enable MoGe metric scaling")
    parser.add_argument("--no-metric-depth", dest="metric_depth", action="store_false")
    parser.set_defaults(metric_depth=True)
    parser.add_argument("--keypoints", default="grid", choices=["aliked", "grid", "none"])
    parser.add_argument("--max-kp", type=int, default=200)
    parser.add_argument("--kp-threshold", type=float, default=0.005)
    parser.add_argument("--estimate-intrinsics", action="store_true", default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    image_paths = list_images(args.images)
    if not image_paths:
        raise SystemExit(f"No images found for: {args.images}")

    cfg = OfflineCreatorConfig(
        model_path=args.model_path,
        output_dir=args.output,
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        device=args.device,
        do_metric_depth=args.metric_depth,
        keypoint_type=args.keypoints,
        max_num_keypoints=args.max_kp,
        keypoint_detection_threshold=args.kp_threshold,
        estimate_camera_params=args.estimate_intrinsics,
        num_loader_workers=args.num_workers,
    )

    creator = OfflineChunkCreator(cfg)
    creator.process_and_save(image_paths)


if __name__ == "__main__":
    main()


