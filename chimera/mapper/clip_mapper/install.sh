#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install clip_mapper..."

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
