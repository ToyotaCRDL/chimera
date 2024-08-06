#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import chimera
from chimera.simulator import Simulator
from pymycobot.mycobot import MyCobot

class MyCobot(Simulator):
    def __init__(self, config=None, device=0, batch_size=1, 
        device_name="/dev/ttyUSB0",
        **kwargs):
        
        if batch_size > 1:
            raise Exception("Cannot set batch_size > 1.")

        self.config = chimera.create_config()
        if config is not None:
            self.config.update(config)
        self.device = device
        
        # Initialize 
        self.mycobot = MyCobot(device_name)

    def step(self, inputs) -> (dict, dict):

        obs = {}
        info = {}

        if "action" in inputs.keys():
            if inputs["action"][0, 0] == 0:
                self.done = True

        if "cmd_ee_pos" in inputs.keys():
            self.mycobot.set_coords()
        
        coords = self.mycobot.get_coords()
        print(coords)
        angles = self.mycobot.get_angles()
        print(angles)

        obs["ee_pos"] = torch.Tensor(coords[:3]).to(self.device).unsqueeze(0)
        obs["ee_rot"] = torch.Tensor(coords[3:]).to(self.device).unsqueeze(0)
        obs["angles"] = torch.Tensor(angles).to(self.device).unsqueeze(0)

        return obs, info

    def reset(self) -> (dict, dict):
        
        obs = {}
        info = {}
        self.done = False

        self.mycobot.send_angles([0, 0, 0, 0, 0, 0], 80)
        coords = self.mycobot.get_coords()
        print(coords)
        angles = self.mycobot.get_angles()
        print(angles)

        obs["ee_pos"] = torch.Tensor(coords[:3]).to(self.device).unsqueeze(0)
        obs["ee_rot"] = torch.Tensor(coords[3:]).to(self.device).unsqueeze(0)
        obs["angles"] = torch.Tensor(angles).to(self.device).unsqueeze(0)
    
        return obs, info
