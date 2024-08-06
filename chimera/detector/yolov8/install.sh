#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install yolov8..."

pip install ultralytics
