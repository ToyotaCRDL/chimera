#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- install Detector..."

args=("$@")
if [ $# -eq 0 ]; then
    # set default module
    args+=("YOLOv8")
fi

for arg in "${args[@]}"; do
    if [ "$arg" = "YOLOv8" ] || [ "$arg" = "yolov8" ] || [ "$arg" = "all" ]; then
        # install yolov8
        bash ./yolov8/install.sh
    fi
    if [ "$arg" = "Detic" ] || [ "$arg" = "detic" ] || [ "$arg" = "all" ]; then
        # install detic
        bash ./detic/install.sh
    fi
    if [ "$arg" = "SegmentAnything" ] || [ "$arg" = "segment_anything" ] || [ "$arg" = "all" ]; then
        # install SegmentAnything
        bash ./segment_anything/install.sh
    fi
done

