#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install habitat..."

# install habitat_sim (use v0.2.4 for VChuken dataset)
conda install habitat-sim=0.2.4 withbullet headless -c conda-forge -c aihabitat # headless

# install habitat_lab
#git clone --branch challenge-2022-branch https://github.com/facebookresearch/habitat-lab.git # use challenge-2022-brach
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git # use stable
cd habitat-lab
pip install -e habitat-lab

