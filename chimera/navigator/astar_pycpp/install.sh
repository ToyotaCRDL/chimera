#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install astar_pycpp..."

git clone https://github.com/srama2512/astar_pycpp.git
cd astar_pycpp
make

