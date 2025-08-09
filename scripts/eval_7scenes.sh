#!/bin/bash

# Configuration
dataset_path="/media/steffen/Data/7scenes/"
output_dir="logs/7scenes"
groundtruth_dir="scripts/groundtruths/7scenes"

# Default parameters
overlap=${1:-5}
chunk_length=${2:-50}

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

# Process each dataset
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
    mkdir -p "$output_dir/$dataset"
    
    # Run SLAM without camera calibration but with visualization
    echo "🔧 Running SLAM with visualization..."
    
    # Build command with optional SIM3 optimization
    cmd="python pi3_slam_online_modular.py \
        --image_dir \"$dataset_name\" \
        --output_path \"$output_dir/$dataset\" \
        --overlap \"$overlap\" \
        --chunk_length \"$chunk_length\" \
        --conf_threshold 0.4 \
        --tum_integer_timestamp \
        --save_tum \
        --no_visualization"
    
    
    eval $cmd
    
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
    traj_file="$output_dir/$dataset/trajectory.tum"
    
    if [ -f "$traj_file" ]; then
        echo "📊 Evaluating trajectory..."
        evo_ape tum "$groundtruth_dir/$dataset.txt" "$traj_file" -as
    else
        echo "⚠️  Trajectory file not found: $traj_file"
    fi

done

echo ""
echo "🎉 7-Scenes evaluation completed!"
echo "📊 Results saved in: $output_dir"
