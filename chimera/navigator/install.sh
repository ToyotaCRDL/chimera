#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install navigation..."

args=("$@")
if [ $# -eq 0 ]; then
    # set default module
    args+=("AStar")
fi

for arg in "${args[@]}"; do
    if [ "$arg" = "AStar" ] || [ "$arg" = "astar" ] || [ "$arg" = "all" ]; then
        # install astar_pycpp
        bash ./astar_pycpp/install.sh
    fi
done



