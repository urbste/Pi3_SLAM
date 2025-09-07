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
import json

from slam.offline_chunk_creator import OfflineChunkCreator, OfflineCreatorConfig
import torch
torch.set_float32_matmul_precision('high')

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1" 
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/home/steffen/.torchinductor_cache"

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
    parser.add_argument("--images", default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_reduced/", help="Folder with images, a glob pattern, or a text file listing image paths")
    parser.add_argument("--model-path", default="/home/steffen/ModelWeights/pi3/model.safetensors", help="Pi3 model identifier or local path for Pi3.from_pretrained")
    parser.add_argument("--output", default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_reduced/chunks", help="Output directory")
    parser.add_argument("--chunk-length", type=int, default=40)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cam-dist-path", type=str, default=None, help="Path to camera calibration file for undistortion")
    parser.add_argument("--metric-depth", action="store_true", help="Enable MoGe metric scaling")
    parser.add_argument("--keypoints", default="grid", choices=["aliked", "grid", "none"])
    parser.add_argument("--max-kp", type=int, default=150)
    parser.add_argument("--kp-threshold", type=float, default=0.005)
    parser.add_argument("--estimate-intrinsics", default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp8", action="store_true", default=False)
    parser.add_argument("--attention-merge", action="store_true")
    parser.add_argument("--merging-ratio", type=float, default=0.6)
    parser.add_argument("--compile-models", action="store_true", help="Compile models")
    parser.add_argument("--pixel-limit", type=int, default=255000, help="Pixel limit for video processing")

    # Optional frame range controls (consistent with online CLI)
    parser.add_argument("--skip-start", type=int, default=0, help="Number of frames to skip from the beginning")
    parser.add_argument("--skip-end", type=int, default=0, help="Number of frames to skip from the end")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every k-th frame (>=1)")
    args = parser.parse_args()

    all_image_paths = list_images(args.images)
    if not all_image_paths:
        raise SystemExit(f"No images found for: {args.images}")

    # Apply frame skipping window
    total_images = len(all_image_paths)
    effective_start = max(0, int(args.skip_start))
    effective_end = total_images - max(0, int(args.skip_end))
    if effective_start >= total_images:
        raise SystemExit(f"Invalid --skip-start {args.skip_start}: exceeds total images {total_images}")
    if effective_end <= effective_start:
        raise SystemExit(f"Invalid frame range after skipping: start {effective_start}, end {effective_end}")

    # Apply stride selection
    stride = max(1, int(args.frame_stride))
    selected_indices = list(range(effective_start, effective_end, stride))
    image_paths = [all_image_paths[i] for i in selected_indices]

    # if chunk length not divisible by 8 fp 8 does not work
    if args.chunk_length % 8 != 0:
        args.fp8 = False
        print(f"Chunk length {args.chunk_length} is not divisible by 8. FP8 does not work. Falling back to float16.")

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
        cam_dist_path=args.cam_dist_path,
        do_fp8=args.fp8,
        do_attention_merge=args.attention_merge,
        merging_ratio=args.merging_ratio,
        compile_models=args.compile_models,
        pixel_limit=args.pixel_limit,
        frame_stride=args.frame_stride,
    )

    # Persist frame selection metadata for later interpolation
    try:
        os.makedirs(args.output, exist_ok=True)
        selection = {
            'original_total_frames': total_images,
            'skip_start': int(args.skip_start),
            'skip_end': int(args.skip_end),
            'frame_stride': stride,
            'effective_start': effective_start,
            'effective_end': effective_end,
            'selected_indices': selected_indices,
            'all_paths': all_image_paths,
        }
        with open(os.path.join(args.output, 'frame_selection.json'), 'w') as f:
            json.dump(selection, f, indent=2)
        print(f"📝 Wrote frame selection metadata: {os.path.join(args.output, 'frame_selection.json')}")
    except Exception as e:
        print(f"⚠️  Failed to write frame selection metadata: {e}")

    creator = OfflineChunkCreator(cfg)
    creator.process_and_save(image_paths)


if __name__ == "__main__":
    main()


