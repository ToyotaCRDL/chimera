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
    if [ "$arg" = "Frontier" ] || [ "$arg" = "frontier" ] || [ "$arg" = "all" ]; then
        # Frontier
        bash ./frontier/install.sh
    fi
    if [ "$arg" = "ExpVS" ] || [ "$arg" = "exp_vs" ] || [ "$arg" = "ExpVSRL" ] || [ "$arg" = "exp_vs_rl" ] || [ "$arg" = "all" ]; then
        # ViewSynthesis
        bash ./exp_vs/install.sh
    fi
done



