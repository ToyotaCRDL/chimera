#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install mycobot..."

pip install pyserial
pip install pymycobot
