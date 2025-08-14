# Pi3_SLAM (Offline Mode)

This repository provides an offline pipeline for chunked reconstruction using [Pi3](https://github.com/yyfz/Pi3) and [MoGe](https://github.com/microsoft/MoGe):

- Create chunks from an image sequence with [Pi3](https://github.com/yyfz/Pi3), scale with [MoGe](https://github.com/microsoft/MoGe), extract keypoints, and save per-chunk data
- Reconstruct and align chunks progressively to build a full scene
- Export final point cloud (PLY) and camera trajectory in TUM format; optional per-chunk SFM/PLY


## Installation

1) Install dependencies
```
pip install -r requirements.txt
```

2) (Optional) Install evaluation tools
```
pip install evo --upgrade
```


## Offline Pipeline

Assumes sequential image data.
The offline pipeline has two steps:

1) Chunk creation ([Pi3](https://github.com/yyfz/Pi3) + [MoGe](https://github.com/microsoft/MoGe) + keypoints)
```
python create_offline_chunks.py \
  --images /path/to/images \
  --model-path yyfz233/Pi3 \
  --output /tmp/pi3_chunks \
  --chunk-length 100 --overlap 10 \
  --device cuda --metric-depth \
  --keypoints grid --max-kp 200 \
  --estimate-intrinsics --num-workers 2
```

Outputs:
- Per-chunk files: `/tmp/pi3_chunks/chunks/chunk_*.pt`
- Metadata: `/tmp/pi3_chunks/chunk_metadata.json`
- Manifest: `/tmp/pi3_chunks/chunks_manifest.json`

2) Reconstruction
```
python reconstruct_offline.py \
  --chunks /tmp/pi3_chunks \
  --output /tmp/pi3_reconstruction \
  --max-observations-per-track 6
```

Outputs:
- `final_points.ply`: Combined point cloud
- `final_camera_poses.ply`: Camera positions as points
- `trajectory_tum.txt`: TUM-format trajectory

Note: Per-chunk SFM/PLY can be enabled with `--save-per-chunk`.


## 7-Scenes Evaluation (Offline)

Use the provided script to run the offline pipeline on 7-Scenes and evaluate with `evo`:
```
bash scripts/eval_7scenes.sh [overlap] [chunk_length]
```

Defaults inside the script: overlap=20, chunk_length=100. The script will:
- Create chunks for each scene
- Reconstruct the scene
- Write `trajectory_tum.txt` and evaluate APE (Sim3 alignment) against ground truth

### Results (APE, RMSE in meters)

| Scene       | RMSE |
|-------------|------:|
| chess       | 0.032 |
| fire        | 0.062 |
| heads       | 0.046 |
| office      | 0.099 |
| pumpkin     | 0.153 |
| redkitchen  | 0.032 |
| stairs      | 0.062 |

** Mean: 0.069 **

Details were computed with `evo_ape tum --align-sim3` using the script’s outputs.


## Notes

- The offline chunk creator reports per-chunk inference time and FPS; the reconstructor reports reconstruction FPS per chunk.
- Chunks store keypoints, per-keypoint 3D, colors, and camera poses for efficient reconstruction.
- `chunk_metadata.json` stores `chunk_length`, `overlap`, and `target_size`; the reconstructor auto-loads this.

## Acknowledgements

This work builds upon and uses ideas and code from the following projects:
- [Pi3](https://github.com/yyfz/Pi3): Scalable permutation-equivariant visual geometry learning. We use Pi3 for chunk-wise geometry (points, local points, camera poses).
- [MoGe](https://github.com/microsoft/MoGe): Monocular geometry with metric scale and sharp details. We use MoGe to recover metric scale for Pi3 reconstructions.

Please refer to their repositories for more details and cite them when appropriate.

## TODO

 - [ ] Add correlation based features refinement
 - [ ] Add localization and reconstruction of another camera
 - [ ] Add gravity residuals
 - [ ] Add GPS residuals