#!/bin/bash

# Configuration defaults
images_path="/home/steffen/Data/GPStrava/TAAWN_TEST_DATA/1/Reference/run1/undist"   # Directory containing images
overall_output_path="logs/gopro/run_all_5_50_lk_chunks_170000"
model_path="/home/steffen/ModelWeights/pi3/model.safetensors"
calib_file=""                                # Optional path to camera calibration JSON

# rm -rf "$overall_output_path"

# Processing defaults
overlap=5
chunk_length=50
mode="offline"                               # offline | online
viz_port=8280
num_workers=10
max_kp=150
metric_depth=true
use_inverse_depth=true
max_observations_per_track=5
keypoints="grid"
use_lk_refinement=false
frame_stride=3


# New features (booleans)
enable_fp8=false
enable_attention_merge=false
merging_ratio="0.8"
compile_models=true
shared_intrinsics=true
pixel_limit=170000

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --images) images_path="$2"; shift 2;;
        --output) overall_output_path="$2"; shift 2;;
        --model-path) model_path="$2"; shift 2;;
        --calib-file) calib_file="$2"; shift 2;;
        --overlap) overlap="$2"; shift 2;;
        --chunk-length) chunk_length="$2"; shift 2;;
        --mode) mode="$2"; shift 2;;
        --viz-port) viz_port="$2"; shift 2;;
        --num-workers) num_workers="$2"; shift 2;;
        --max-kp) max_kp="$2"; shift 2;;
        --metric-depth) metric_depth="$2"; shift 2;;             # true|false
        --use-inverse-depth) use_inverse_depth="$2"; shift 2;;   # true|false
        --max-observations-per-track) max_observations_per_track="$2"; shift 2;;
        --use-lk-refinement) use_lk_refinement="$2"; shift 2;;   # true|false
        --enable-fp8) enable_fp8="$2"; shift 2;;                 # true|false
        --enable-attention-merge) enable_attention_merge="$2"; shift 2;; # true|false
        --merging-ratio) merging_ratio="$2"; shift 2;;
        --compile-models) compile_models="$2"; shift 2;;
        --shared-intrinsics) shared_intrinsics="$2"; shift 2;;
        --keypoints) keypoints="$2"; shift 2;;
        --pixel-limit) pixel_limit="$2"; shift 2;;
        --frame-stride) frame_stride="$2"; shift 2;;
        *) shift;;
    esac
done

# Validate inputs
if [ ! -d "$images_path" ]; then
    echo "❌ Images directory does not exist: $images_path"
    exit 1
fi

mkdir -p "$overall_output_path"

echo "🚀 Starting GoPro processing with Pi3SLAM"
echo "📂 Images: $images_path"
echo "💾 Output: $overall_output_path"
echo "🧭 Mode: $mode"
echo "🔧 Overlap: $overlap, Chunk length: $chunk_length"
echo "🤖 Model: $model_path"
echo "🧪 FP8: $enable_fp8, Attention merge: $enable_attention_merge, Merge ratio: $merging_ratio"
echo "============================================================"

if [[ "$mode" == "offline" ]]; then
    chunks_out="$overall_output_path"
    recon_out="$overall_output_path/reconstruction"
    timing_out="$overall_output_path/timings.txt"

    # Clean previous reconstruction
    if [ -d "$recon_out" ]; then
        echo "🧹 Removing old reconstruction folder: $recon_out"
        rm -rf "$recon_out"
    fi
    mkdir -p "$recon_out"

    echo "🔧 Creating offline chunks..."
    t0=$(date +%s)
    python create_offline_chunks.py \
        --images "$images_path" \
        --model-path "$model_path" \
        --output "$chunks_out" \
        --chunk-length "$chunk_length" \
        --overlap "$overlap" \
        --device cuda \
        $( [ -n "$calib_file" ] && echo "--cam-dist-path \"$calib_file\"" ) \
        $( $metric_depth && echo "--metric-depth" ) \
        --keypoints "$keypoints" \
        --max-kp "$max_kp" \
        --num-workers "$num_workers" \
        $( $enable_fp8 && echo "--fp8" ) \
        $( $enable_attention_merge && echo "--attention-merge --merging-ratio $merging_ratio" ) \
        $( $compile_models && echo "--compile-models" ) \
        --pixel-limit "$pixel_limit" \
        --frame-stride "$frame_stride"
    t1=$(date +%s)
    chunk_secs=$((t1 - t0))

    echo "🏗️  Reconstructing from chunks..."
    t2=$(date +%s)
    python reconstruct_offline.py \
        --chunks "$chunks_out" \
        --output "$recon_out" \
        --max-observations-per-track "$max_observations_per_track" \
        $( $use_inverse_depth && echo "--use-inverse-depth" ) \
        $( $shared_intrinsics && echo "--shared-intrinsics" ) \
        $( $use_lk_refinement && echo "--use-lk-refinement" ) \
        --num-workers "$num_workers"
    t3=$(date +%s)
    recon_secs=$((t3 - t2))

    {
        echo "chunk_creation_seconds: $chunk_secs"
        echo "reconstruction_seconds: $recon_secs"
    } > "$timing_out"
    echo "⏱️  Timings written to: $timing_out"

    echo "✅ Completed (offline)"
else
    # Online pipeline
    online_out="$overall_output_path/online"
    timing_out="$overall_output_path/online_timing.txt"

    if [ -d "$online_out" ]; then
        echo "🧹 Removing old online output folder: $online_out"
        rm -rf "$online_out"
    fi
    mkdir -p "$online_out"

    echo "🚦 Starting online reconstruction with visualization..."
    t0=$(date +%s)
    python pi3_slam_online_modular.py \
        --image_dir "$images_path" \
        $( [ -n "$calib_file" ] && echo "--cam_dist_path \"$calib_file\"" ) \
        --model_path "$model_path" \
        --device cuda \
        --chunk_length "$chunk_length" \
        --overlap "$overlap" \
        --keypoint_type "$keypoints" \
        --max_num_keypoints "$max_kp" \
        --max_observations_per_track "$max_observations_per_track" \
        --do_metric_depth \
        --use_inverse_depth \
        --output_path "$online_out" \
        --save_tum \
        --viz_port "$viz_port" \
        --num-workers "$num_workers" \
        --frame-stride "$frame_stride"
    t1=$(date +%s)
    online_secs=$((t1 - t0))

    {
        echo "online_seconds: $online_secs"
    } > "$timing_out"
    echo "⏱️  Online timing written to: $timing_out"

    echo "✅ Completed (online)"
fi

 echo ""
 echo "🎉 GoPro processing completed!"
 echo "📊 Results saved in: $overall_output_path"
