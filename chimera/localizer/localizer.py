#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import abc

class Localizer(metaclass=abc.ABCMeta):
    """
    This class is an abstract class which provides interface of Localizer.
    """

    @abc.abstractmethod
    def reset(self):
        """
        Abstract method to reset Localizer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def track(self, inputs) -> dict:
        """
        Abstract method to track position using new observations.

        Args:
            inputs (dict): Inputs with new observations.

        Returns:
            dict: Outputs (position (and/or position2d), rotation (and/or rotation2d), map2d).
        """
        raise NotImplementedError()

# For backward compatibility
def create_localization(config=None, name="DroidSLAM", device=0, batch_size=1, **kwargs):
    return create_localizer(config, name, device, batch_size, **kwargs)

def create_localizer(config=None, name="DroidSLAM", device=0, batch_size=1, **kwargs):
    """
    Create an Localizer.

    Args:
        config (dict) The config parameters for the environment.: 
        name (str): The name of localization method (e.g., `DroidSLAM`)
        device (int): The number of cuda device.
        batch_size (int): The batch size.

    Returns:
        Localizer: The instance of the defined Localizer.
    """

    print("Localizer = " + name)
    if name == "droidslam" or name == "DroidSLAM":
        from chimera.localizer.droidslam import DroidSLAM
        return DroidSLAM(config, device, batch_size, **kwargs)
    else:
        raise NotImplementedError("There is no Localizer named:" + name)

