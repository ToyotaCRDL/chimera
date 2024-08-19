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
done
