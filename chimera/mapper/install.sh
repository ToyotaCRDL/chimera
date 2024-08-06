#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- install mapper..."

args=("$@")
if [ $# -eq 0 ]; then
    # set default module
    args+=("mapper2d" "clip_mapper" "l2m")
fi

for arg in "${args[@]}"; do
    if [ "$arg" = "mapper2d" ] || [ "$arg" = "all" ]; then
        # install mapper2d
        bash ./mapper2d/install.sh
    fi

    if [ "$arg" = "clip_mapper" ] || [ "$arg" = "all" ]; then
        # install clip_mapper
        bash ./clip_mapper/install.sh
    fi

    if [ "$arg" = "l2m" ] || [ "$arg" = "all" ]; then
        # install l2m
        bash ./l2m/install.sh
    fi

    if [ "$arg" = "vlmaps" ] || [ "$arg" = "all" ]; then
        # install vlmaps
        bash ./vlmaps/install.sh
    fi
done

