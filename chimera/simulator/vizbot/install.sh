#!/bin/bash

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
echo "- - install vizbot..."

# Install ROS tools
pip install trollius rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_pkg
pip install git+https://github.com/catkin/catkin_tools
