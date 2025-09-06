"""
CLI to reconstruct a sequence from saved offline chunks.

Example:
  python reconstruct_offline.py \
    --chunks /path/to/output_chunks/chunks \
    --output /path/to/output_chunks/reconstruction \
    --chunk-length 80 --overlap 10
"""

from __future__ import annotations

import argparse
import os

from slam.offline_reconstructor import OfflineReconstructor


def main():
    parser = argparse.ArgumentParser(description="Reconstruct from saved PI3 chunks")
    parser.add_argument("--chunks", default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_small/chunks", help="Directory containing chunk_*.pt files")
    parser.add_argument("--output", default="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist_small/reconstruction", help="Directory to write reconstructions (sfm/ply)")
    parser.add_argument("--chunk-length", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--max-observations-per-track", type=int, default=5)
    parser.add_argument("--save-per-chunk", action="store_true", help="Save per-chunk .sfm/.ply files as well")
    parser.add_argument("--use-inverse-depth", action="store_true", help="Use inverse depth parametrization")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--use-lk-refinement", action="store_true", help="Use Lucas-Kanade refinement")
    parser.add_argument("--debug-lk-refinement", action="store_true", help="Debug Lucas-Kanade refinement")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    recon = OfflineReconstructor(
        chunk_dir=args.chunks,
        output_dir=args.output,
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        max_observations_per_track=args.max_observations_per_track,
        save_per_chunk=args.save_per_chunk,
        use_inverse_depth=args.use_inverse_depth,
        num_workers=args.num_workers,
        use_lk_refinement=args.use_lk_refinement,
        debug_lk_refinement=args.debug_lk_refinement,
    )
    recon.run()


if __name__ == "__main__":
    main()


