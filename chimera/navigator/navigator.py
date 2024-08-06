#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import abc

class Navigator(metaclass=abc.ABCMeta):
    """
    This class is an abstract class which provides interface of Navigator.
    """

    @abc.abstractmethod
    def reset(self):
        """
        Abstract method to reset Navigator.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def act(self, inputs) -> dict:
        """
        Abstract method to add new observations.

        Args:
            inputs (dict): Inputs with new observations.

        Returns:
            dict: Outputs (action).
        """
        raise NotImplementedError()

# For backward compatibility
def create_navigation(config=None, name="Astar", device=0, batch_size=1, **kwargs):
    return create_navigator(config, name, device, batch_size, **kwargs)

def create_navigator(config=None, name="Astar", device=0, batch_size=1, **kwargs):
    """
    Create an Navigator

    Args:
        config (dict) The config parameters for the environment.: 
        name (str): The name of navigation method (e.g., `Astar`)
        device (int): The number of cuda device.
        batch_size (int): The batch size.

    Returns:
        Navigator: The instance of the defined Navigator.
    """
    print("Navigator = " + name)
    if name == "astar" or name == "Astar":
        from chimera.navigator.astar_pycpp import Astar
        return Astar(config, device, batch_size, **kwargs)
    else:
        raise NotImplementedError("There is no Navigator named:" + name)


