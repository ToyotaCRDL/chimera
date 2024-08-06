#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install stablediffusion..."

pip install transformers diffusers invisible-watermark
pip install accelerate
