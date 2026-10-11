#!/usr/bin/env bash
# Download the parts of KITTI 3D object detection needed for training/validation.
#   bash scripts/download_kitti.sh [target_dir]      (default: /content/kitti)
# The velodyne zip is ~29 GB; only training/ is extracted. Needs ~60 GB free disk during extraction.
set -euo pipefail

DIR="${1:-/content/kitti}"
B="https://s3.eu-central-1.amazonaws.com/avg-kitti"
G="https://raw.githubusercontent.com/open-mmlab/OpenPCDet/master/data/kitti/ImageSets"

mkdir -p "$DIR/ImageSets"
cd "$DIR"

wget -nc -q "$B/data_object_calib.zip"
wget -nc -q "$B/data_object_label_2.zip"
wget -nc    "$B/data_object_velodyne.zip"

unzip -n -q data_object_calib.zip    'training/*'
unzip -n -q data_object_label_2.zip  'training/*'
unzip -n -q data_object_velodyne.zip 'training/*'
rm -f data_object_velodyne.zip

wget -nc -q -O ImageSets/train.txt "$G/train.txt"
wget -nc -q -O ImageSets/val.txt   "$G/val.txt"

echo "velodyne scans: $(ls training/velodyne | wc -l)"
echo "train / val ids: $(wc -l < ImageSets/train.txt) / $(wc -l < ImageSets/val.txt)"
