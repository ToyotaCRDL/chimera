#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import chimera
from chimera.navigator import Navigator
from chimera.navigator.astar_pycpp.astar_pycpp import pyastar

import math
import numpy as np
import torch
import torch.nn.functional as F

class Astar(Navigator):
    def __init__(self, config, device=0, batch_size=1, 
        margin=0.15,
        goal_radius=0.15,
        **kwargs):
        self.config = config
        self.map_size = self.config["map2d"]["size"]
        self.map_scale = self.config["map2d"]["scale"]
        self.agent_action_type = self.config["agent"]["action_type"]
        if self.agent_action_type == "discrete":
            self.agent_forward_step_size = self.config["agent"]["forward_step_size"]
            self.agent_turn_angle = self.config["agent"]["turn_angle"] * np.pi / 180.0
        elif self.agent_action_type == "continuous":
            self.agent_max_velocity = self.config["agent"]["max_velocity"]
            self.agent_max_angular_velocity = self.config["agent"]["max_angular_velocity"]
            self.sampling_interval = self.config["sampling_interval"]
        else:
            raise Exception("Unknown action_type: " + self.agent_action_type)
        self.agent_radius = self.config["agent"]["radius"]
        self.device = device
        self.batch_size = batch_size
        self.margin = margin
        
        # Filter
        filter_radius = math.ceil((self.agent_radius + self.margin) / self.map_scale)
        filter_size = 2 * filter_radius + 1
        y, x = np.ogrid[-filter_radius:filter_radius+1, -filter_radius:filter_radius+1]
        mask = x ** 2 + y ** 2 < (self.agent_radius + self.margin) / self.map_scale
        filter_weight = np.zeros((1, 1, filter_size, filter_size), dtype=np.float32)
        filter_weight[:, :, mask] = 1
        filter_weight = torch.from_numpy(filter_weight).to(self.device)
        def circular_sum(input_tensor):
            padding = filter_radius
            output_tensor = F.conv2d(input_tensor, filter_weight.to(input_tensor.device), padding=padding)
            return output_tensor
        self.circular_sum = circular_sum

        # Grid
        grid_x = torch.Tensor(range(self.map_size)).to(self.device).view(1, 1, -1, 1).repeat(1, 1, 1, self.map_size)
        grid_y = torch.Tensor(range(self.map_size)).to(self.device).view(1, 1, 1, -1).repeat(1, 1, self.map_size, 1)
        self.grid = torch.cat([grid_x, grid_y], dim=1)

        self.goal_radius = goal_radius
        self.paths = []
        for b in range(self.batch_size):
            self.paths.append([])

    def reset(self):
        self.paths = []
        for b in range(self.batch_size):
            self.paths.append([])

    def act(self, inputs, **kwargs) -> dict:
        map2d = inputs["map2d"].clone()
        pos2d = inputs["position2d"].clone()
        rot2d = inputs["rotation2d"].clone()
        map_pos = torch.round(pos2d / self.map_scale + self.map_size / 2).to(torch.int64)
        paths = []
        obs = map2d == 0
        obsmap = torch.zeros_like(map2d).to(self.device)
        obsmap[obs] = 1
        obsmap = self.circular_sum(obsmap)
        obsmap = obsmap.permute(0, 1, 3, 2)
        paths = -1 * torch.ones(self.batch_size, 100, 2).to(self.device).to(torch.int64)
        gmap = torch.zeros(self.map_size, self.map_size).to(self.device)
        for b in range(self.batch_size):
        
            goal2d_xy = inputs["goal2d_xy"]
            if self.batch_size > 1 or isinstance(goal2d_xy, list):
                goal2d_xy = inputs["goal2d_xy"][b]
                
            map_goal = torch.round(goal2d_xy / self.map_scale + self.map_size / 2).to(torch.int64)
        
            if map_goal.dim() < 2:
                map_goal = map_goal.unsqueeze(0)
            
            gmap.fill_(0)
            free_area = int(self.goal_radius / self.map_scale) // 2
            goals = []
            for n in range(map_goal.size(0)):
                obsmap[b, 0, 
                    map_goal[n, 0]-free_area:map_goal[n, 0]+free_area+1, 
                    map_goal[n, 1]-free_area:map_goal[n, 1]+free_area+1] = 0
                gmap[map_goal[n, 0], map_goal[n, 1]] = 1
            goals.append(map_goal[:, 0].cpu().numpy().tolist())
            goals.append(map_goal[:, 1].cpu().numpy().tolist())

            path = pyastar.multi_goal_weighted_astar_path(obsmap[b, 0].cpu().numpy(), gmap.cpu().numpy(), map_pos[b].cpu().numpy(), goals, allow_diagonal=True)
            
            path = torch.from_numpy(path).to(self.device)
            length = min(path.shape[0],paths.shape[1])
            if length > 0:
                paths[b, :length] = path[:length]
                paths[b, length:] = path[-1]

        # calc_action
        if self.agent_action_type == "discrete":
            forecast = max(1, int(self.agent_forward_step_size / self.map_scale))
            target = paths[:, forecast]
            direction = target - map_pos
            phi = -torch.atan2(direction[..., 1], direction[..., 0])
            action = torch.ones(self.batch_size, 1).to(self.device).to(torch.int64) # Default 1:Move Forward
            dang = phi - rot2d
            dang[dang > np.pi] -= 2 * np.pi
            dang[dang < -np.pi] += 2 * np.pi
            action[dang < -self.agent_turn_angle / 2] = 3 # Turn Right
            action[dang > self.agent_turn_angle / 2] = 2 # Turn Left
            path_length = torch.zeros(self.batch_size, 1).to(self.device)
            for n in range(paths.size(1) - 1):
                p1 = (paths[:, n, :] - self.map_size / 2) * self.map_scale
                p2 = (paths[:, n + 1, :] - self.map_size / 2) * self.map_scale
                path_length += torch.norm(p2 - p1, dim=1)
            action[path_length <= self.goal_radius] = 0 # Done
            if "prev_action" in inputs.keys():
                prev_action = inputs["prev_action"].clone()
                action[action==2 and prev_action == 3] = 1 # Default 1:Move Forward
                action[action==3 and prev_action == 2] = 1 # Default 1:Move Forward
                if "collision" in inputs.keys():
                    collision = inputs["collision"].clone()
                    action[(prev_action==1) and (collision==1) and (dang < 0)] = 3 # Turn Right
                    action[(prev_action==1) and (collision==1) and (dang >= 0)] = 2 # Turn Left
            action[paths[:, 0, 0] == -1, 0] = -1

            outputs = {
                "path": paths,
                "action": action,
            }
        elif self.agent_action_type == "continuous":
            forecast = max(1, int(self.agent_max_velocity * self.sampling_interval / self.map_scale))
            target = paths[:, forecast]
            direction = target.to(torch.float32) - map_pos.to(torch.float32)
            phi = -torch.atan2(direction[..., 1], direction[..., 0])
            cmd_vel = torch.zeros(self.batch_size, 2).to(self.device)
            dang = phi - rot2d
            dang[dang > np.pi] -= 2 * np.pi
            dang[dang < -np.pi] += 2 * np.pi
            dist = torch.norm(direction, dim=1)
            vel = dist / self.sampling_interval
            vel[vel > self.agent_max_velocity] = self.agent_max_velocity
            vel = vel * torch.cos(dang)
            vel[vel < 0.05] = 0
            ang_vel = -2 * dang / self.sampling_interval
            ang_vel[ang_vel > self.agent_max_angular_velocity] = self.agent_max_angular_velocity
            ang_vel[ang_vel < -self.agent_max_angular_velocity] = -self.agent_max_angular_velocity
            cmd_vel[:, 0] = vel
            cmd_vel[:, 1] = ang_vel

            action = torch.ones(self.batch_size, 1).to(self.device).to(torch.int64)

            refine_goal = (paths[:, -1, :] - self.map_size / 2) * self.map_scale
            goal_dist = torch.norm(refine_goal - pos2d, dim=1)
            action[goal_dist <= self.goal_radius] = 0 # Done

            action[paths[:, 0, 0] == -1, 0] = -1 # Fail

            outputs = {
                "path": paths,
                "action": action,
                "cmd_vel": cmd_vel,
            }
        return outputs

