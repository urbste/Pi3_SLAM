#!/bin/bash

# Configuration
dataset_path="/media/steffen/Data/7scenes/"
output_dir="logs/7scenes"
groundtruth_dir="scripts/groundtruths/7scenes"

# Default parameters
overlap=${1:-5}
chunk_length=${2:-30}

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
echo "============================================================"

###############################################################################
# Offline pipeline: create chunks then reconstruct (no visualization)
for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path$dataset/seq-01/color/"

    if [ ! -d "$dataset_name" ]; then
        echo "⚠️  Dataset directory not found: $dataset_name"
        continue
    fi

    echo ""
    echo "🎯 Processing dataset: $dataset"
    echo "📂 Dataset path: $dataset_name"

    # Create dataset-specific output directories
    chunks_out="$output_dir/$dataset"
    recon_out="$output_dir/$dataset/reconstruction"
    mkdir -p "$chunks_out"
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
        --max-kp 200 \
        --estimate-intrinsics \
        --num-workers 2 

    echo "🏗️  Reconstructing from chunks..."
    python reconstruct_offline.py \
        --chunks "$chunks_out" \
        --output "$recon_out" \
        --max-observations-per-track 7

    echo "✅ Completed processing for $dataset"
done

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
    
    # Check if trajectory files exist
    traj_file="$output_dir/$dataset/reconstruction/trajectory_tum.txt"
    
    if [ -f "$traj_file" ]; then
        echo "📊 Evaluating trajectory..."
        plot_out="$output_dir/$dataset/reconstruction/evo_ape.png"
        evo_ape tum "$groundtruth_dir/$dataset.txt" "$traj_file" -as --plot --plot_mode xyz --save_plot "$plot_out"
        echo "🖼️  Saved APE plot to: $plot_out"
    else
        echo "⚠️  Trajectory file not found: $traj_file"
    fi

done

echo ""
echo "🎉 7-Scenes evaluation completed!"
echo "📊 Results saved in: $output_dir"
