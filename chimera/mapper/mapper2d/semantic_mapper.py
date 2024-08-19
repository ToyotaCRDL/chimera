#!/usr/bin/env python3
# Copyright (c) Toyota Central R&D Labs., Inc.
# All rights reserved.

import chimera
from chimera.mapper import Mapper
import os
import math
import numpy as np
import torch
import torch.nn as nn
import pytorch3d.transforms as p3dt
import quaternion
import cv2

class SemanticMapper(Mapper):
    def __init__(self, config, device=0, batch_size=1, **kwargs):

        self.device = device
        self.batch_size = batch_size
        self.map_scale = config["map2d"]["scale"]
        self.map_size = config["map2d"]["size"]
        self.sensor_height = config["depth"]["position"][1]
        self.agent_height = config["agent"]["height"]
        
        # parameters about depth
        self.depth_width = config["depth"]["width"]
        self.depth_height = config["depth"]["height"]
        self.min_depth = config["depth"]["min_depth"]
        self.max_depth = config["depth"]["max_depth"]
        self.depth_hfov = config["depth"]["hfov"]

        # semantic category
        if "category" in kwargs.keys():
            self.category = kwargs["category"]
        elif "objects" in config.keys() and "names" in config["objects"].keys():
            self.category = config["objects"]["names"]
        else:
            self.category = {}
        self.nc = len(self.category.keys())

        if "conf_thresh" in kwargs.keys():
            self.conf_thresh = kwargs["conf_thresh"]
        elif "objects" in config.keys() and "conf_thresh" in config["objects"].keys():
            self.conf_thresh = config["objects"]["conf_thresh"]
        else:
            self.conf_thresh = 0.7

        # camera intrinsics
        if "intrinsics" in config["depth"].keys():
            fx = config["depth"]["intrinsics"][0]
            cx = config["depth"]["intrinsics"][2]
            fy = config["depth"]["intrinsics"][4]
            cy = config["depth"]["intrinsics"][5]
            K = np.array([[fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]], dtype="float32")
        else:
            f = 0.5 * self.depth_width / math.tan(0.5 * self.depth_hfov / 180.0 * math.pi)
            cx = 0.5 * self.depth_width
            cy = 0.5 * self.depth_height
            K = np.array([[f, 0.0, cx, 0.0],
                [0.0, f, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]], dtype="float32")
        inv_K = np.linalg.pinv(K)
        self.K = torch.from_numpy(K).to(self.device)
        self.inv_K = torch.from_numpy(inv_K).to(self.device)

        # pix coords
        meshgrid = np.meshgrid(range(self.depth_width), range(self.depth_height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = torch.from_numpy(self.id_coords).to(self.device)
        self.pix_coords = torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0)
        self.pix_coords = self.pix_coords + 0.5
        ones = torch.ones(1, self.depth_height * self.depth_width).to(self.device)
        self.pix_coords = torch.cat([self.pix_coords, ones], 0)        
        self.camera_coords = self.inv_K[:3, :3] @ self.pix_coords

        self.map = torch.zeros(self.batch_size, self.nc, self.map_size, self.map_size).to(self.device)

        grid_x = torch.Tensor(range(self.map_size)).to(self.device).view(1, 1, -1, 1).repeat(1, 1, 1, self.map_size)
        grid_y = torch.Tensor(range(self.map_size)).to(self.device).view(1, 1, 1, -1).repeat(1, 1, self.map_size, 1)
        self.grid = torch.cat([grid_x, grid_y], dim=1).repeat(self.batch_size, 1, 1, 1)

        conv = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1)
        weight = torch.zeros(self.nc, self.nc, 3, 3)
        weight[range(self.nc), range(self.nc), :, :] = 1    
        conv.weight = nn.Parameter(weight)
        conv = conv.to(self.device)

        def mean_filter(x):
            threshold = conv.kernel_size[0] * conv.kernel_size[1] / 2.0
            x = conv(x)
            x = torch.where(x > threshold, torch.ones_like(x), torch.zeros_like(x)) 
            return x

        self.filter = mean_filter

    def change_category(self, category):

        self.category = category
        self.nc = len(self.category.keys())
        
        conv = nn.Conv2d(self.nc, self.nc, kernel_size=3, stride=1, padding=1)
        weight = torch.zeros(self.nc, self.nc, 3, 3)
        weight[range(self.nc), range(self.nc), :, :] = 1    
        conv.weight = nn.Parameter(weight)
        conv = conv.to(self.device)

        def mean_filter(x):
            threshold = conv.kernel_size[0] * conv.kernel_size[1] / 2.0
            x = conv(x)
            x = torch.where(x > threshold, torch.ones_like(x), torch.zeros_like(x)) 
            return x

        self.filter = mean_filter

        self.map = torch.zeros(self.batch_size, self.nc, self.map_size, self.map_size).to(self.device)

    def reset(self):

        self.map = torch.zeros(self.batch_size, self.nc, self.map_size, self.map_size).to(self.device)

    def add(self, inputs) -> dict:

        segments = torch.zeros(self.batch_size, self.nc, self.depth_height, self.depth_width).to(self.device)
        if "segments" in inputs.keys():
            seg = inputs["segments"]
            for k in seg.keys():
                for c in range(self.nc):
                    if chimera.check_object_category(k, self.category[c]):
                        for b in range(self.batch_size):
                            segments[b, c, seg[k][b, 0] > self.conf_thresh] = 1

        elif "objects" in inputs.keys():
            obj = inputs["objects"]
            if "names" in obj.keys():
                if len(obj["names"]) != self.nc:
                    map_old = self.map
                    self.change_category(obj["names"])
                    self.map[:, :map_old.size(1)] = map_old
                    segments = torch.zeros(self.batch_size, self.nc, self.depth_height, self.depth_width).to(self.device)

            for b in range(self.batch_size):
                objects = obj["boxes"][b]
                for n in range(objects.shape[0]):
                    if objects[n][4].item() > self.conf_thresh:
                        c = int(objects[n][5].item())
                        segments[b, c, obj["masks"][b, n] == 1] = 1

        segments = segments.view(self.batch_size, self.nc, -1)

        depth = inputs["depth"].clone()
        depth = depth.view(self.batch_size, 1, -1)
        depth_valid = torch.logical_and(depth[:, 0, :,] > 0, depth[:, 0, :] < 1)
        depth = self.min_depth + depth * (self.max_depth - self.min_depth)
        local_points = depth * self.camera_coords
        ones = torch.ones(self.batch_size, 1, self.depth_height * self.depth_width).to(self.device)
        local_points = torch.cat([local_points, ones], 1)        

        if "position" in inputs.keys() and "rotation" in inputs.keys():
            pos = inputs["position"].clone()
            pos[:, 1] = 0
            T = torch.zeros(self.batch_size, 4, 4).to(self.device)
            T[:, :3, :3] = p3dt.quaternion_to_matrix(inputs["rotation"])
            T[:, :3, 3] = pos
            T[:, 3, 3] = 1
            R = torch.diag(torch.Tensor([1, -1, -1, 1]).to(self.device))
            points = (T @ R) @ local_points
        elif "position2d" in inputs.keys() and "rotation2d" in inputs.keys():
            pos2d = inputs["position2d"].clone()
            rot2d = inputs["rotation2d"].clone()
            zerorot2d = torch.zeros_like(rot2d)
            axis_angle = torch.cat([zerorot2d, rot2d, zerorot2d], dim=1)
            T = torch.zeros(self.batch_size, 4, 4).to(self.device)
            T[:, :3, :3] = p3dt.axis_angle_to_matrix(axis_angle)
            T[:, 0, 3] = pos2d[:, 1]
            T[:, 2, 3] = -pos2d[:, 0]
            T[:, 3, 3] = 1
            R = torch.diag(torch.Tensor([1, -1, -1, 1]).to(self.device))
            points = (T @ R) @ local_points
        else:
            raise Exception("There are no position (position2d) and rotation (rotation2d).")

        for b in range(self.batch_size):
            for c in range(self.nc):
                sem_points = points[b, :, segments[b, c] == 1] / self.map_scale
                if sem_points.numel() > 0:
                    sem_points = sem_points[[0, 2]]
                    sem_points[1] = -sem_points[1]
                    sem_points += self.map_size / 2
                    sem_points = sem_points.to(torch.int64)

                    self.map[b, c, sem_points[0, :], sem_points[1, :]] = 1

        self.map = self.filter(self.map)

        outputs = {
            "semmap2d": self.map,
        }

        return outputs

    def find(self, inputs):
        
        objectgoal = inputs["objectgoal"]

        pos2d = inputs["position2d"] / self.map_scale + self.map_size / 2
        pos2d = pos2d[:, [1, 0]]
        distance = torch.norm(self.grid.permute(0, 2, 3, 1) - pos2d, dim=3)

        obj_points = []
        for b in range(self.batch_size):
            channel = -1
            for k in self.category.keys():
                if chimera.check_object_category(objectgoal[b], self.category[k]):
                    channel = k
                    break
            if channel == -1:
                obj_points.append(None)
                continue            
            mask = (self.map[b, channel] == 1)
            if self.map[b, channel, mask].numel() > 0:
                value = self.map[b, channel]
                value_dist = value / (distance[b] + 1)
                maxvalue_pos = torch.argmax(value_dist.view(-1), dim=0)
                obj_point = torch.zeros(2).to(self.device)
                obj_point[0] = (torch.fmod(maxvalue_pos, self.map_size) - self.map_size / 2) * self.map_scale
                obj_point[1] = (torch.floor_divide(maxvalue_pos, self.map_size) - self.map_size / 2) * self.map_scale
                obj_points.append(obj_point.unsqueeze(0))
            else:
                obj_points.append(None)

        outputs = {
            "goal2d_xy": obj_points,
        }

        return outputs

    def find_nearest(self, objectgoal, position2d):

        objectgoal = inputs["objectgoal"]
        position2d = inputs["position2d"]
        pos2d = position2d / self.map_scale + self.map_size / 2
        pos2d = pos2d[:, [1, 0]]

        distance = torch.norm(self.grid.permute(0, 2, 3, 1) - pos2d, dim=3)

        obj_points = []
        for b in range(self.batch_size):
            channel = -1
            for k in self.category.keys():
                if chimera.check_object_category(objectgoal[b], self.category[k]):
                    channel = k
                    break
            if channel == -1:
                obj_points.append(None)
                continue

            mask = (self.map[b, channel] == 1)
            if self.map[b, channel, mask].numel() > 0:
                value = self.map[b, channel]
                value_dist = value / (distance[b] + 1)
                maxvalue_pos = torch.argmax(value_dist.view(-1), dim=0)
                obj_point = torch.zeros(2).to(self.device)
                obj_point[0] = (torch.fmod(maxvalue_pos, self.map_size) - self.map_size / 2) * self.map_scale
                obj_point[1] = (torch.floor_divide(maxvalue_pos, self.map_size) - self.map_size / 2) * self.map_scale
                obj_points.append(obj_point)
            else:
                obj_points.append(None)

        return obj_points

        
