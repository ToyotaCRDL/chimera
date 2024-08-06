#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.


def create_config():
    """
    Create a config.

    Args:
    
    Returns:
        dict: A default config.
    """

    dict_conf = {
        "agent": {
            "forward_step_size": 0.25,
            "turn_angle": 30,
            "radius": 0.18,
            "height": 0.88,
        },
        "rgb": {
            "width": 640,
            "height": 480,
            "hfov": 79,
            "position": [0.0, 0.88, 0.0],
        },
        "depth": {
            "width": 640,
            "height": 480,
            "hfov": 79,
            "min_depth": 0.5,
            "max_depth": 5.0,
            "position": [0.0, 0.88, 0.0],
        },
        "map2d": {
            "scale": 0.05,
            "size": 2400,
        }
    }

    return dict_conf

