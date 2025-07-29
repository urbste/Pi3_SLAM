#!/bin/bash

# Configuration
dataset_path="/media/steffen/Data/Euroc/"
calib_file="example/euroc_cam0_calib.json"
output_dir="logs/euroc"
groundtruth_dir="scripts/groundtruths/euroc"

# Create output directories
mkdir -p "$output_dir/calib"

datasets=(
    MH_01_easy
)

echo "🚀 Starting EuroC dataset evaluation with Pi3SLAM"
echo "📁 Dataset path: $dataset_path"
echo "📷 Calibration file: $calib_file"
echo "📊 Output directory: $output_dir"
echo "============================================================"

# Process each dataset
for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path$dataset/"
    
    if [ ! -d "$dataset_name" ]; then
        echo "⚠️  Dataset directory not found: $dataset_name"
        continue
    fi
    
    echo ""
    echo "🎯 Processing dataset: $dataset"
    echo "📂 Dataset path: $dataset_name"
    
    # Create dataset-specific output directories
    mkdir -p "$output_dir/calib/$dataset"
    
    # Run SLAM with calibration
    echo "🔧 Running SLAM with camera calibration..."
    python pi3_slam_online_rerun_modular.py \
        --image_dir "$dataset_name"/mav0/cam0/data \
        --cam_dist_path "$calib_file" \
        --output_path "$output_dir/calib/$dataset" \
        --save_tum \
        --overlap 20 \
        --chunk_length 100 \
        --conf_threshold 0.5 \
        --skip_start 1000 \
        --no_visualization
 
    
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
    calib_traj="$output_dir/calib/$dataset/trajectory.tum"
    
    if [ -f "$calib_traj" ]; then
        echo "📊 Evaluating with calibration..."
        evo_ape tum "$groundtruth_dir/$dataset.txt" "$calib_traj" -as
    else
        echo "⚠️  Calibrated trajectory file not found: $calib_traj"
    fi

done

echo ""
echo "🎉 EuroC evaluation completed!"