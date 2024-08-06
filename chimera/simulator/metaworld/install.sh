#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install metaworld..."

# install metaworld
pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld
