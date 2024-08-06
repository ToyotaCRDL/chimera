#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- install localization..."

args=("$@")
if [ $# -eq 0 ]; then
    # set default module
    args+=("droidslam")
fi

for arg in "${args[@]}"; do
    if [ "$arg" = "droidslam" ] || [ "$arg" = "all" ]; then
        # install DROID-SLAM
        bash droidslam/install.sh
    fi
    if [ "$arg" = "orbslam3" ] || [ "$arg" = "all" ]; then
        # install ORB-SLAM3
        bash orbslam3/install.sh
    fi
done
