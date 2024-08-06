#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT

args=("$@")
if [ $# -eq 1 ] && [ "$args" = "all" ]; then
    args+=("all")
fi
if [ $# -eq 0 ]; then
    # set default module
    args+=("all")
fi
bash ./chimera/install.sh "${args[@]}"

if [ "$args" = "all" ] || [ "$args" = "main" ]; then
    pip install -r requirements.txt
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install pytorch3d -c pytorch3d
    pip install -e .
fi
