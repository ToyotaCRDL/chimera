#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install segment_anything..."

pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install git+https://github.com/facebookresearch/segment-anything.git
if  [ ! -f "sam_vit_h_4b8939.pth" ]; then
    wget --no-check-certificate https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

