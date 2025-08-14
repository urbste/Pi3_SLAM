#!/bin/bash

# Configuration
dataset_path="/media/steffen/Data/Euroc/"
output_dir="logs/euroc"
groundtruth_dir="scripts/groundtruths/euroc"
calib_file="example/euroc_cam0_calib.json"

# Defaults (can be overridden by CLI flags)
overlap=10
chunk_length=100
mode="offline"    # offline | online
viz_port_base=8180 # base port for online visualization

# Parse args (named flags)
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
        --calib-file)
            calib_file="$2"; shift 2;;
        *)
            shift;;
    esac
done

# Create output directories
mkdir -p "$output_dir"

datasets=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    MH_05_difficult
)

# Per-dataset start frame index (edit values as needed)
# as there is a loooong static start in the dataset
declare -A START_IDX
START_IDX=(
    [MH_01_easy]=885
    [MH_02_easy]=922
    [MH_03_medium]=388
    [MH_04_difficult]=415
    [MH_05_difficult]=425
)

echo "🚀 Starting EuroC dataset evaluation with Pi3SLAM"
echo "📁 Dataset path: $dataset_path"
echo "📊 Output directory: $output_dir"
echo "📷 Calibration file: $calib_file"
echo "🔧 Overlap: $overlap, Chunk length: $chunk_length"
echo "🧭 Mode: $mode"
echo "============================================================"

# Run reconstruction per mode
if [[ "$mode" == "offline" ]]; then
    # Offline pipeline: create chunks then reconstruct (no visualization)
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path$dataset/"
        
        if [ ! -d "$dataset_name" ]; then
            echo "⚠️  Dataset directory not found: $dataset_name"
            continue
        fi
        
        echo ""
        echo "🎯 Processing dataset (offline): $dataset"
        echo "📂 Dataset path: $dataset_name"
        start_idx=${START_IDX[$dataset]:-0}
        echo "⏩ Start frame index: $start_idx"
        
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
            --images "$dataset_name"/mav0/cam0/data \
            --cam-dist-path "$calib_file" \
            --model-path "yyfz233/Pi3" \
            --output "$chunks_out" \
            --chunk-length "$chunk_length" \
            --overlap "$overlap" \
            --device cuda \
            --metric-depth \
            --keypoints grid \
            --max-kp 400 \
            --skip-start "$start_idx" \
            --estimate-intrinsics \
            --num-workers 2

        echo "🏗️  Reconstructing from chunks..."
        python reconstruct_offline.py \
            --chunks "$chunks_out" \
            --output "$recon_out" \
            --max-observations-per-track 7 \
            --use-inverse-depth

        echo "✅ Completed processing (offline) for $dataset"
    done
else
    # Online pipeline: run SLAM with visualization and save TUM for eval
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path$dataset/"
        
        if [ ! -d "$dataset_name" ]; then
            echo "⚠️  Dataset directory not found: $dataset_name"
            continue
        fi
        
        echo ""
        echo "🎯 Processing dataset (online): $dataset"
        echo "📂 Dataset path: $dataset_name"
        start_idx=${START_IDX[$dataset]:-0}
        echo "⏩ Start frame index: $start_idx"

        online_out="$output_dir/$dataset/online"
        # Clean previous online outputs if present
        if [ -d "$online_out" ]; then
            echo "🧹 Removing old online reconstruction folder: $online_out"
            rm -rf "$online_out"
        fi
        mkdir -p "$online_out"

        echo "🚦 Starting online reconstruction with visualization..."
        python pi3_slam_online_modular.py \
            --image_dir "$dataset_name"/mav0/cam0/data \
            --cam_dist_path "$calib_file" \
            --model_path "yyfz233/Pi3" \
            --device cuda \
            --chunk_length "$chunk_length" \
            --overlap "$overlap" \
            --keypoint_type grid \
            --max_num_keypoints 400 \
            --max_observations_per_track 7 \
            --do_metric_depth \
            --use_inverse_depth \
            --skip_start "$start_idx" \
            --output_path "$online_out" \
            --save_tum \
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
echo "🎉 EuroC evaluation completed!"