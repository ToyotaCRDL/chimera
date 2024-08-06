#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install openai..."

# install openai
pip install openai
