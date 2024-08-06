#!/usr/bin/env python3
# Copyright (c) Toyota Central R&D Labs., Inc.
# All rights reserved.

from chimera.mapper import Mapper
import os
import math
import numpy as np
import torch
import torch.nn as nn
import pytorch3d.transforms as p3dt
import quaternion
import cv2

class RGBMapper(Mapper):
    def __init__(self, config, device=0, batch_size=1, map_scale=0.05, map_size=2400, **kwargs):

        self.device = device
        self.batch_size = batch_size
        self.map_scale = config["map2d"]["scale"]
        self.map_size = config["map2d"]["size"]
        self.sensor_height = config["depth"]["position"][1]
        self.agent_height = config["agent"]["height"]
        self.height_thresh_min = -self.sensor_height + 0.3
        self.height_thresh_max = max(self.agent_height, self.sensor_height) - self.sensor_height + 0.1
        if "height_thresh_min" in kwargs.keys():
            self.height_thresh_min = kwargs["height_thresh_min"]
        if "height_thresh_max" in kwargs.keys():
            self.height_thresh_max = kwargs["height_thresh_max"]
        
        # parameters about depth
        self.depth_width = config["depth"]["width"]
        self.depth_height = config["depth"]["height"]
        self.min_depth = config["depth"]["min_depth"]
        self.max_depth = config["depth"]["max_depth"]
        self.depth_hfov = config["depth"]["hfov"]

        # check parameter of rgb
        if self.depth_width != config["rgb"]["width"]:
            raise Exception("width of rgb is not same as that of depth.")
        if self.depth_height != config["rgb"]["height"]:
            raise Exception("height of rgb is not same as that of depth.")
        if self.depth_hfov != config["rgb"]["hfov"]:
            raise Exception("hfov of rgb is not same as that of depth.")

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

        self.map = torch.zeros(self.batch_size, 3, self.map_size, self.map_size).to(self.device)        

    def reset(self):

        self.map = torch.zeros(self.batch_size, 3, self.map_size, self.map_size).to(self.device)

    def add(self, inputs) -> dict:

        rgb = inputs["rgb"].clone()
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

        height_valid = points[:, 1, :] < self.height_thresh_max
        valid = torch.logical_and(depth_valid, height_valid)
        rgb = rgb.view(self.batch_size, 3, -1)      

        for b in range(self.batch_size):

            map_points = points[b] / self.map_scale
            map_points = map_points[[0, 2]]
            map_points[1] = -map_points[1]
            map_points += self.map_size / 2
            map_points = map_points.to(torch.int64)
            
            self.map[b, :, map_points[0, valid[b]], map_points[1, valid[b]]] = rgb[b, :, valid[b]]

        outputs = {
            "rgbmap2d": self.map,
        }

        return outputs

