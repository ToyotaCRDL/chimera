#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import abc

class Mapper(metaclass=abc.ABCMeta):
    """
    This class is an abstract class which provides interface of mapper.
    """

    @abc.abstractmethod
    def reset(self):
        """
        Abstract method to reset mapper.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def add(self, inputs) -> dict:
        """
        Abstract method to add new observations.

        Args:
            inputs (dict): Inputs with new observations.

        Returns:
            dict: Outputs (map2d).
        """
        raise NotImplementedError()

def create_mapper(config=None, name="DepthAndTrajectoryMapper", device=0, batch_size=1, **kwargs):
    """
    Create a Mapper.

    Args:
        config (dict) The config parameters for the environment.: 
        name (str): The name of mapper method (e.g., `DepthAndTrajectoryMapper`)
        device (int): The number of cuda device.
        batch_size (int): The batch size.

    Returns:
        Mapper: The instance of the defined Mapper.
    """
    print("Mapper = " + name)
    if name == "depth_mapper" or name == "DepthMapper":
        from chimera.mapper.mapper2d import DepthMapper
        return DepthMapper(config, device, batch_size, **kwargs)
    elif name == "trajectory_mapper" or name == "TrajectoryMapper":
        from chimera.mapper.mapper2d import TrajectoryMapper
        return TrajectoryMapper(config, device, batch_size, **kwargs)
    elif name == "depth_and_trajectory_mapper" or name == "DepthAndTrajectoryMapper":
        from chimera.mapper.mapper2d import DepthAndTrajectoryMapper
        return DepthAndTrajectoryMapper(config, device, batch_size, **kwargs)
    elif name == "rgb_mapper" or name == "RGBMapper":
        from chimera.mapper.mapper2d import RGBMapper
        return RGBMapper(config, device, batch_size, **kwargs)
    elif name == "semantic_mapper" or name == "SemanticMapper":
        from chimera.mapper.mapper2d import SemanticMapper
        return SemanticMapper(config, device, batch_size, **kwargs)
    elif name == "clip_mapper" or name == "CLIPMapper":
        from chimera.mapper.clip_mapper import CLIPMapper
        return CLIPMapper(config, device, batch_size, **kwargs)
    elif name == "vlmap" or name == "VLMap":
        from chimera.mapper.vlmaps import VLMap
        return VLMap(config, device, batch_size, **kwargs)
    elif name == "l2m" or name == "L2M" or name == "LanguageToMap":
        from chimera.mapper.l2m import L2M
        return L2M(config, device, batch_size, **kwargs)
    else:
        raise NotImplementedError("There is no Mapper named:" + name)

