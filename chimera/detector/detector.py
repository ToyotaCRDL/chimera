#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import abc

class Detector(metaclass=abc.ABCMeta):
    """
    This class is an abstract class which provides interface of Detector.
    """

    @abc.abstractmethod
    def reset(self):
        """
        Abstract method to reset Detector.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def detect(self, inputs) -> dict:
        """
        Abstract method to detect objects.

        Args:
            inputs (dict): Inputs of observations.

        Returns:
            dict: Outputs (objects).
        """
        raise NotImplementedError()

# For backward compatibility
def create_object_detection(config=None, name="YOLOv8", device=0, batch_size=1, **kwargs):
    return create_detector(config, name, device, batch_size, **kwargs)

# For backward compatibility
def create_semantic_segmentation(config=None, name="SegmentAnything", device=0, batch_size=1, **kwargs):
    return create_detector(config, name, device, batch_size, **kwargs)

def create_detector(config=None, name="YOLOv8", device=0, batch_size=1, **kwargs):
    """
    Create a Detector.

    Args:
        config (dict) The config parameters for the environment.: 
        name (str): The name of detection method (e.g., `YOLOv8`)
        device (int): The number of cuda device.
        batch_size (int): The batch size.

    Returns:
        Detector: The instance of the defined Detector.
    """
    print("Detector = " + name)
    if name == "detic" or name == "Detic":
        from chimera.detector.detic import Detic
        return Detic(config, device, batch_size, **kwargs)
    elif name == "yolov8" or name == "YOLOv8":
        from chimera.detector.yolov8 import YOLOv8
        return YOLOv8(config, device, batch_size, **kwargs)
    elif name == "segment_anything" or name == "SegmentAnything":
        from chimera.detector.segment_anything import SegmentAnything
        return SegmentAnything(config, device, batch_size, **kwargs)
    else:
        raise NotImplementedError("There is no Detector named:" + name)

