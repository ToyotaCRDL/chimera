#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "install chimera..."

args=("$@")
if [ $# -eq 1 ] && [ "$args" = "all" ]; then
    args+=("all")
fi
if [ $# -eq 0 ]; then
    # set default module
    args+=("all")
fi
sub_args=("${args[@]:1}")

# Simulator
if [ "$args" = "simulator" ] || [ "$args" = "Simulator" ] || [ "$args" = "all" ]; then
    bash ./simulator/install.sh "${sub_args[@]}"
fi

# Navigator
if [ "$args" = "navigator" ] || [ "$args" = "Navigator" ] || [ "$args" = "all" ]; then
    bash ./navigator/install.sh "${sub_args[@]}"
fi

# Mapper
if [ "$args" = "mapper" ] || [ "$args" = "Mapper" ] || [ "$args" = "all" ]; then
    bash ./mapper/install.sh "${sub_args[@]}"
fi

# Detector
if [ "$args" = "detector" ] || [ "$args" = "Detector" ] || [ "$args" = "all" ]; then
    bash ./detector/install.sh "${sub_args[@]}"
fi

# Localizer
if [ "$args" = "localizer" ] || [ "$args" = "Localizer" ] || [ "$args" = "all" ]; then
    bash ./localizer/install.sh "${sub_args[@]}"
fi

# Generator
if [ "$args" = "generator" ] || [ "$args" = "Generator" ]  || [ "$args" = "all" ]; then
    bash ./generator/install.sh "${sub_args[@]}"
fi

