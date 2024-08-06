#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- install simulators..."

args=("$@")
if [ $# -eq 0 ]; then
    # set default module
    args+=("habitat")
fi

for arg in "${args[@]}"; do
    if [ "$arg" = "habitat" ] || [ "$arg" = "all" ]; then
        # install habitat
        bash ./habitat/install.sh
    fi
    if [ "$arg" = "gibson" ] || [ "$arg" = "all" ]; then
        # install gibson
        bash ./gibson/install.sh
    fi
    if [ "$arg" = "metaworld" ] || [ "$arg" = "all" ]; then
        # install metaworld
        bash ./metaworld/install.sh
    fi
    if [ "$arg" = "vizbot" ] || [ "$arg" = "all" ]; then
        # install vizbot
        bash ./vizbot/install.sh
    fi
done

