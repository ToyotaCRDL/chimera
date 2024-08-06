#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)

cd $CURRENT

git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM
git apply ../droidslam.patch

cd $CURRENT
pip install evo --upgrade --no-binary evo
pip install gdown
conda install pytorch-scatter -c pyg
conda install suitesparse

cd DROID-SLAM
python setup.py install
gdown https://drive.google.com/uc?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh

