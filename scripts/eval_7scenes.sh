#!/bin/bash

# Configuration
dataset_path="/media/steffen/Data/7scenes/"
output_dir="logs/7scenes"
groundtruth_dir="scripts/groundtruths/7scenes"

# Defaults
overlap=5
chunk_length=50
mode="offline"    # offline | online
viz_port_base=8080 # used for online mode

# Parse args (named flags override defaults)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --overlap)
            overlap="$2"; shift 2;;
        --chunk-length)
            chunk_length="$2"; shift 2;;
        --mode)
            mode="$2"; shift 2;;
        --viz-port-base)
            viz_port_base="$2"; shift 2;;
        --dataset-path)
            dataset_path="$2"; shift 2;;
        --output-dir)
            output_dir="$2"; shift 2;;
        --groundtruth-dir)
            groundtruth_dir="$2"; shift 2;;
        *)
            # Ignore unknown positional args for backward compatibility
            shift;;
    esac
done

# Create output directories
mkdir -p "$output_dir"

datasets=(
    chess
    fire
    heads
    office
    pumpkin
    redkitchen
    stairs
)

echo "🚀 Starting 7-Scenes dataset evaluation with Pi3SLAM"
echo "📁 Dataset path: $dataset_path"
echo "📊 Output directory: $output_dir"
echo "🔧 Overlap: $overlap, Chunk length: $chunk_length"
echo "🧭 Mode: $mode"
echo "============================================================"

###############################################################################
# Run reconstruction per mode
if [[ "$mode" == "offline" ]]; then
    # Offline pipeline: create chunks then reconstruct (no visualization)
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path$dataset/seq-01/color/"

        if [ ! -d "$dataset_name" ]; then
            echo "⚠️  Dataset directory not found: $dataset_name"
            continue
        fi

        echo ""
        echo "🎯 Processing dataset (offline): $dataset"
        echo "📂 Dataset path: $dataset_name"

        # Create dataset-specific output directories
        chunks_out="$output_dir/$dataset"
        recon_out="$output_dir/$dataset/reconstruction"
        mkdir -p "$chunks_out"

        # Clean previous reconstruction outputs if present
        if [ -d "$recon_out" ]; then
            echo "🧹 Removing old reconstruction folder: $recon_out"
            rm -rf "$recon_out"
        fi
        mkdir -p "$recon_out"

        echo "🔧 Creating offline chunks..."
        python create_offline_chunks.py \
            --images "$dataset_name" \
            --model-path "yyfz233/Pi3" \
            --output "$chunks_out" \
            --chunk-length "$chunk_length" \
            --overlap "$overlap" \
            --device cuda \
            --metric-depth \
            --keypoints grid \
            --max-kp 400 \
            --estimate-intrinsics \
            --num-workers 2 

        echo "🏗️  Reconstructing from chunks..."
        python reconstruct_offline.py \
            --chunks "$chunks_out" \
            --output "$recon_out" \
            --max-observations-per-track 10

        echo "✅ Completed processing (offline) for $dataset"
    done
else
    # Online pipeline: run SLAM with visualization and save TUM for eval
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path$dataset/seq-01/color/"

        if [ ! -d "$dataset_name" ]; then
            echo "⚠️  Dataset directory not found: $dataset_name"
            continue
        fi

        echo ""
        echo "🎯 Processing dataset (online): $dataset"
        echo "📂 Dataset path: $dataset_name"

        online_out="$output_dir/$dataset/online"
        # Clean previous online outputs if present
        if [ -d "$online_out" ]; then
            echo "🧹 Removing old online reconstruction folder: $online_out"
            rm -rf "$online_out"
        fi
        mkdir -p "$online_out"

        echo "🚦 Starting online reconstruction with visualization..."
        python pi3_slam_online_modular.py \
            --image_dir "$dataset_name" \
            --model_path "yyfz233/Pi3" \
            --device cuda \
            --chunk_length "$chunk_length" \
            --overlap "$overlap" \
            --keypoint_type grid \
            --max_num_keypoints 400 \
            --max_observations_per_track 10 \
            --do_metric_depth \
            --output_path "$online_out" \
            --save_tum \
            --tum_integer_timestamp \
            --viz_port "$viz_port_base"

        echo "✅ Completed processing (online) for $dataset"
    done
fi

echo ""
echo "📊 Running evaluation metrics..."
echo "============================================================"

# Evaluate each dataset
for dataset in ${datasets[@]}; do
    echo ""
    echo "📈 Evaluating dataset: $dataset"
    
    # Check if groundtruth exists
    if [ ! -f "$groundtruth_dir/$dataset.txt" ]; then
        echo "⚠️  Groundtruth file not found: $groundtruth_dir/$dataset.txt"
        continue
    fi
    
    # Select trajectory file based on mode
    if [[ "$mode" == "offline" ]]; then
        traj_file="$output_dir/$dataset/reconstruction/trajectory_tum.txt"
        plot_out="$output_dir/$dataset/reconstruction/evo_ape.png"
    else
        traj_file="$output_dir/$dataset/online/trajectory.tum"
        plot_out="$output_dir/$dataset/online/evo_ape.png"
    fi

    if [ -f "$traj_file" ]; then
        echo "📊 Evaluating trajectory..."
        evo_ape tum "$groundtruth_dir/$dataset.txt" "$traj_file" -as --plot --plot_mode xyz --save_plot "$plot_out"
        echo "🖼️  Saved APE plot to: $plot_out"
    else
        echo "⚠️  Trajectory file not found: $traj_file"
    fi

done

echo ""
echo "🎉 7-Scenes evaluation completed!"
echo "📊 Results saved in: $output_dir"
