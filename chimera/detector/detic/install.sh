#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install detic..."

pip install git+https://github.com/facebookresearch/detectron2.git

cd $CURRENT
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip install -r requirements.txt

# Download model
mkdir -p models
cd models
if  [ ! -f "BoxSup-C2_LCOCO_CLIP_SwinB_896b32_4x.pth" ]; then
    wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/BoxSup-C2_LCOCO_CLIP_SwinB_896b32_4x.pth
fi
if  [ ! -f "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth" ]; then
    wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
fi
