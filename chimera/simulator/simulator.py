#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import abc

class Simulator(metaclass=abc.ABCMeta):
    """
    This class is an abstract class which provides interface of simulators.
    """

    @abc.abstractmethod
    def get_config(self) -> dict:
        """
        Abstract method to get config of simulator.

        Returns:
            dict: The config of simulator.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def num_episodes(self) -> int:
        """
        Abstract method to get a number of episodes.

        Returns:
            int: The number of episodes.
        """
        raise NotImplementedError()



    @abc.abstractmethod
    def step(self, inputs=dict) -> (dict, dict):
        """
        Abstract method to proceed the simulation.

        Args:
            inputs (dict): action(Tensor(batch, 1)).

        Returns:
            dict: Observations from simulation.
            dict: Information provided from simulation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> (dict, dict):
        """
        Abstract method to reset simulation.

        Returns:
            dict: Observations from simulation.
            dict: Information provided from simulation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_episode_over(self) -> bool:
        """
        Returns a flag which represents current episode is done or not.

        Returns:
            bool: Flag represents current episode is done or not.
        """
        raise NotImplementedError()

def create_simulator(name="Habitat", config=None, device=0, task="objectnav", dataset="hm3d", split="val_mini", batch_size=1, **kwargs):
    """
    Create a Simulator.
    
    Args:
        name (str): The name of simulator (e.g., `Habitat`).
        config (OmegaConf): The config of evaluation
        device (int): The number of cuda device.
        task (str): The name of task (e.g., `objectnav`).
        dataset (str): The name of dataset (e.g., `hm3d`).
        split (str): The split of dataset (e.g., `val_mini`).
        batch_size (int): The number of parallel processed simulators.

    Returns:
        Simulator: The instance of the defined simulator.
    """
    print("Simulator = " + name)
    if name == "habitat" or name == "Habitat":
        from chimera.simulator.habitat import Habitat
        return Habitat(config=config, device=device, task=task, dataset=dataset, split=split, batch_size=batch_size, **kwargs)
    elif name == "gibson" or name == "Gibson":
        from chimera.simulator.gibson import Gibson
        return Gibson(config=config, device=device, task=task, dataset=dataset, split=split, batch_size=batch_size, **kwargs)
    elif name == "metaworld" or name == "MetaWorld":
        from chimera.simulator.metaworld import MetaWorld
        return MetaWorld(config=config, device=device, task=task, dataset=dataset, split=split, batch_size=batch_size, **kwargs)
    elif name == "vln_ce" or name == "VLN-CE":
        from chimera.simulator.vln_ce import VLNCE
        return VLNCE(config=config, device=device, task=task, dataset=dataset, split=split, batch_size=batch_size, **kwargs)
    elif name == "vizbot" or name == "Vizbot":
        from chimera.simulator.vizbot import Vizbot
        return Vizbot(config=config, device=device, task=task, dataset=dataset, split=split, batch_size=batch_size, **kwargs)
    else:
        raise NotImplementedError("There is no Simulator named:" + name)

