#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- install image_generation..."

args=("$@")
if [ $# -eq 0 ]; then
    # set default module
    args+=("openai" "stablediffusion")
fi

for arg in "${args[@]}"; do
    if [ "$arg" = "openai" ] || [ "$arg" = "OpenAI" ] || [ "$arg" = "all" ]; then
        # install openai
        bash ./openai/install.sh
    fi
    if [ "$arg" = "google_generativeai" ] || [ "$arg" = "GoogleGenerativeAI" ] || [ "$arg" = "all" ]; then
        # install unilm
        bash ./google_generativeai/install.sh
    fi
    if [ "$arg" = "stablediffusion" ] || [ "$arg" = "StableDiffusion" ] || [ "$arg" = "all" ]; then
        # install stable diffusion
        bash ./stablediffusion/install.sh
    fi
done
