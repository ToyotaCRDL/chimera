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

    if [ "$arg" = "llama" ] || [ "$arg" = "Llama" ] || [ "$arg" = "all" ]; then
        # install llama
        bash ./llama/install.sh
    fi

    if [ "$arg" = "unilm" ] || [ "$arg" = "UniLM" ] || [ "$arg" = "all" ]; then
        # install unilm
        bash ./unilm/install.sh
    fi
    if [ "$arg" = "stablediffusion" ] || [ "$arg" = "StableDiffusion" ] || [ "$arg" = "all" ]; then
        # install stable diffusion
        bash ./stablediffusion/install.sh
    fi
done