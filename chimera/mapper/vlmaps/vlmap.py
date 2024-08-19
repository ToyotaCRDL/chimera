#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import os
import sys
import argparse
import math
import chimera
from chimera.mapper import Mapper
vlmaps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlmaps")
sys.path.append(vlmaps_dir)

import numpy as np
import torch
import pytorch3d.transforms as p3dt

from vlmaps.map.vlmap import VLMap as vlm;
from vlmaps.map.vlmap_builder import VLMapBuilder
from omegaconf import DictConfig, OmegaConf

from vlmaps.utils.lseg_utils import get_lseg_feat
from vlmaps.utils.mapping_utils import (
    transform_pc,
    base_pos2grid_id_3d,
    project_point,
    get_sim_cam_mat,
)
from vlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
import clip
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_2d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)
from vlmaps.utils.clip_utils import get_lseg_score
import cv2

class VLMap(Mapper):
    def __init__(self, config, device=0, batch_size=1, **kwargs):
        
        # camera intrinsics
        self.min_depth = config["depth"]["min_depth"]
        self.max_depth = config["depth"]["max_depth"]
        self.depth_width = config["depth"]["width"]
        self.depth_height = config["depth"]["height"]
        self.depth_hfov = config["depth"]["hfov"]
        f = 0.5 * self.depth_width / math.tan(0.5 * self.depth_hfov / 180.0 * math.pi)
        cx = 0.5 * self.depth_width
        cy = 0.5 * self.depth_height
        cam_calib_mat = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]

        config_dict = {
            "map_type": "vlmap",
            "pose_info": {
                "pose_type": "mobile_base",
                "camera_height": config["depth"]["position"][1],
                "base2cam_rot": [1, 0, 0, 0, -1, 0, 0, 0, -1],
                "base_forward_axis": [0, 0, -1],
                "base_left_axis": [-1, 0, 0],
                "base_up_axis": [0, 1, 0],
                },
            "cam_calib_mat": cam_calib_mat,
            "grid_size": config["map2d"]["size"],
            "cell_size": config["map2d"]["scale"],
            "depth_sample_rate": 100,
            "dilate_iter": 3,
            "gaussian_sigma": 1.0,
            "customize_obstacle_map": False,
            "min_depth": config["depth"]["min_depth"],
            "max_depth": config["depth"]["max_depth"],
            }
        map_config = OmegaConf.create(config_dict)
        self.device = device
        self.batch_size = batch_size
        self.gs = map_config["grid_size"]
        self.cs = map_config["cell_size"]
        self.vlmap = vlm(map_config, data_dir="maps")
        self.vlmap._init_clip()
        self.builder = VLMapBuilderOnline(self.vlmap.map_config, self.vlmap.base2cam_tf, self.vlmap.base_transform)

    def create_map_from_data(self, data_dir="maps"):

        self.vlmap.create_map(data_dir)

    def load(self, data_dir="maps"):

        self.vlmap.load_map(data_dir)
        self.vlmap.generate_obstacle_map()

    def reset(self):
        self.builder.reset()

    def add(self, inputs):

        if "position" in inputs.keys() and "rotation" in inputs.keys():
            pos = inputs["position"].clone()
            pos[:, 1] = 0
            T = torch.zeros(self.batch_size, 4, 4).to(self.device)
            T[:, :3, :3] = p3dt.quaternion_to_matrix(inputs["rotation"])
            T[:, :3, 3] = pos
            T[:, 3, 3] = 1
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

        rgbs = inputs["rgb"]
        depths = inputs["depth"] * (self.max_depth - self.min_depth) + self.min_depth
        for b in range(self.batch_size):
            rgb = rgbs[b].permute(1, 2, 0).cpu().numpy()
            depth = depths[b, 0].cpu().numpy()
            pose = T[b].cpu().numpy()
            (
                grid_feat, 
                grid_pos, 
                weight, 
                occupied_ids, 
                mapped_iter_set_list, 
                grid_rgb
            ) = self.builder.add(rgb, depth, pose)
            self.vlmap.grid_feat = grid_feat
            self.vlmap.grid_pos = grid_pos
            self.vlmap.weight = weight
            self.vlmap.occupied_ids = occupied_ids
            self.vlmap.mapped_iter_list = mapped_iter_set_list
            self.vlmap.grid_rgb = grid_rgb

        self.vlmap.generate_obstacle_map()

        outputs = {}

        return outputs

    def find(self, inputs):
        return find_nearest(inputs)

    def find_nearest(self, inputs):

        batch_goals2d = []
        objgoals = inputs["objectgoal"]
        self.vlmap.init_categories(objgoals)

        pos2d = inputs["position2d"].clone()
        p = torch.zeros(pos2d.shape[0], 2).to(self.device)
        p[:, 0] = (-pos2d[:, 0] / self.cs + self.gs / 2) 
        p[:, 1] = pos2d[:, 1] / self.cs + self.gs / 2

        for b in range(self.batch_size):

            curr_pos = p[b].cpu().tolist()
            row, col = self.vlmap.get_nearest_pos(curr_pos, objgoals[b])
            pos = [row, col]
            goals2d = torch.zeros(1, 2).to(self.device)
            goals2d[:, 0] = - self.cs * (pos[0] - self.gs / 2)
            goals2d[:, 1] = self.cs * (pos[1] - self.gs / 2)

            batch_goals2d.append(goals2d)
            
        outputs = {
            "goal2d_xy": batch_goals2d,
        }

        return outputs

    def find_at_1(self, inputs):
        batch_goals2d = []
        objgoals = inputs["objectgoal"]
        self.vlmap.init_categories(objgoals)

        for b in range(self.batch_size):

            scores_mat = get_lseg_score(
                self.vlmap.clip_model,
                [objgoals[b]],
                self.vlmap.grid_feat,
                self.vlmap.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other

            norm_score = scores_mat[:, 0] / (scores_mat[:, 1] + 1e-1)
            max_pos = np.argmax(norm_score, axis=0)
            row, col, h = self.vlmap.grid_pos[max_pos]            

            pos = [row, col]
            goals2d = torch.zeros(1, 2).to(self.device)
            goals2d[:, 0] = - self.cs * (pos[0] - self.gs / 2)
            goals2d[:, 1] = self.cs * (pos[1] - self.gs / 2)

            batch_goals2d.append(goals2d)
            
        outputs = {
            "goal2d_xy": batch_goals2d,
        }

        return outputs
        

    def get_rgbmap2d(self):
        rgb_2d = pool_3d_rgb_to_2d(self.vlmap.grid_rgb, self.vlmap.grid_pos, self.gs)
        return torch.flip(torch.from_numpy(rgb_2d).to(self.device).permute(2, 1, 0).unsqueeze(0), dims=[3])

    def get_heatmap2d(self, objgoal):
        self.vlmap.init_categories([objgoal])
        mask = self.vlmap.index_map(objgoal, with_init_cat=True)
        mask_2d = pool_3d_label_to_2d(mask, self.vlmap.grid_pos, self.gs)
        return torch.flip(torch.from_numpy(mask_2d).to(self.device).permute(1, 0).unsqueeze(0).unsqueeze(0), dims=[3])
        



class VLMapBuilderOnline(VLMapBuilder):
    def __init__(self, map_config, base2cam_tf, base_transform):
        self.map_config = map_config
        self.base2cam_tf = base2cam_tf
        self.base_transform = base_transform
        self.camera_height = self.map_config.pose_info.camera_height
        self.cs = self.map_config.cell_size
        self.gs = self.map_config.grid_size
        self.min_depth = self.map_config.min_depth
        self.max_depth = self.map_config.max_depth
        self.depth_sample_rate = self.map_config.depth_sample_rate
        self.lseg_model, self.lseg_transform, self.crop_size, self.base_size, self.norm_mean, self.norm_std = self._init_lseg()
        self.calib_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))
        self.init_base_tf = self.base_transform @ np.eye(4) @ np.linalg.inv(self.base_transform)
        self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        self.init_cam_tf = self.init_base_tf @ self.base2cam_tf
        self.inv_init_cam_tf = np.linalg.inv(self.init_cam_tf)
        self.reset()

    def reset(self):
        (        
            self.vh,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
            self.mapped_iter_set,
            self.max_id,
        ) = self._init_map(self.camera_height, self.cs, self.gs, "")
        self.cv_map = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.height_map = -100 * np.ones((self.gs, self.gs), dtype=np.float32)

    def add(self, rgb, depth, pose):

        base_pose = self.base_transform @ pose @ np.linalg.inv(self.base_transform)        
        tf = self.inv_init_base_tf @ base_pose

        pix_feats = get_lseg_feat(
            self.lseg_model, rgb, ["example"], self.lseg_transform, self.device, self.crop_size, self.base_size, self.norm_mean, self.norm_std)
        pix_feats_intr = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

        pc = self._backproject_depth(depth, self.calib_mat, self.depth_sample_rate, min_depth=self.min_depth, max_depth=self.max_depth)
        pc_transform = tf @ self.base_transform @ self.base2cam_tf
        pc_global = transform_pc(pc, pc_transform)  # (3, N)

        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p[0], p[1], p[2])
            if self._out_of_range(row, col, height, self.gs, self.vh):
                continue

            px, py, pz = project_point(self.calib_mat, p_local)
            rgb_v = rgb[py, px, :]
            px, py, pz = project_point(pix_feats_intr, p_local)

            if height > self.height_map[row, col]:
                self.height_map[row, col] = height
                self.cv_map[row, col, :] = rgb_v

            # when the max_id exceeds the reserved size,
            # double the grid_feat, grid_pos, weight, grid_rgb lengths
            if self.max_id >= self.grid_feat.shape[0]:
                self._reserve_map_space(self.grid_feat, self.grid_pos, self.weight, self.grid_rgb)

            # apply the distance weighting according to
            # ConceptFusion https://arxiv.org/pdf/2302.07241.pdf Sec. 4.1, Feature fusion
            radial_dist_sq = np.sum(np.square(p_local))
            sigma_sq = 0.6
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

            # update map features
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
                occupied_id = self.occupied_ids[row, col, height]
                if occupied_id == -1:
                    self.occupied_ids[row, col, height] = self.max_id
                    self.grid_feat[self.max_id] = feat.flatten() * alpha
                    self.grid_rgb[self.max_id] = rgb_v
                    self.weight[self.max_id] += alpha
                    self.grid_pos[self.max_id] = [row, col, height]
                    self.max_id += 1
                else:
                    self.grid_feat[occupied_id] = (
                        self.grid_feat[occupied_id] * self.weight[occupied_id] + feat.flatten() * alpha
                    ) / (self.weight[occupied_id] + alpha)
                    self.grid_rgb[occupied_id] = (self.grid_rgb[occupied_id] * self.weight[occupied_id] + rgb_v * alpha) / (
                        self.weight[occupied_id] + alpha
                    )
                    self.weight[occupied_id] += alpha

        return self.grid_feat, self.grid_pos, self.weight, self.occupied_ids, list(self.mapped_iter_set), self.grid_rgb
