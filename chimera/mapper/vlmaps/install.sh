#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install vlmaps..."

git clone https://github.com/vlmaps/vlmaps.git
pip install -r requirements.txt

# Download LSeg model
$DOWNLOAD_DIR="vlmaps/vlmaps/lseg/checkpoints"
if [ ! -d $DOWNLOAD_DIR/demo_e200.ckpt ]
    gdown https://drive.google.com/uc?id=1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb
    mv demo_e200.ckpt $DOWNLOAD_DIR/
fi
