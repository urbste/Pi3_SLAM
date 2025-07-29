#!/bin/bash

dest="/media/steffen/Data/7scenes"
mkdir -p "$dest"

urls=(
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip"
)

for url in "${urls[@]}"; do
    file_name=$(basename "$url")
    # echo "Downloading $file_name..."
    # wget "$url" -O "$dest/$file_name"
    # echo "Unzipping $file_name..."
    # unzip "$dest/$file_name" -d "$dest"
    # unzip "$dest/${file_name%.*}/seq-01" -d "$dest/${file_name%.*}"
    
    # Create color subdirectory for each scene
    scene_name="${file_name%.*}"
    color_dir="$dest/$scene_name/seq-01/color"
    mkdir -p "$color_dir"
    
    # Copy all .color.png files to the color subdirectory
    if [ -d "$dest/$scene_name/seq-01" ]; then
        echo "Copying .color.png files for $scene_name..."
        mv "$dest/$scene_name/seq-01/"*.color.png "$color_dir/" 2>/dev/null || echo "No .color.png files found for $scene_name"
    else
        echo "Directory not found: $dest/$scene_name/seq-01"
    fi
done