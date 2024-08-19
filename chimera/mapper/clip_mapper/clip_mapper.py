#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import chimera
from chimera.mapper import Mapper

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import pytorch3d.transforms as p3dt
import quaternion
import cv2
import glob
from PIL import Image
import torchvision.transforms as transforms

import clip


class CLIPMapper(Mapper):
    def __init__(self, config, device=0, batch_size=1, clip_model="ViT-L/14", scales=[-1, 0, 1], point_merge=True, dist_merge_thresh=1000, sim_merge_thresh=0.9 **kwargs):

        self.device = device
        self.batch_size = batch_size
        self.scales = scales
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

        # camera intrinsics
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

        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.tokenizer = clip.tokenize

        self.nc = self.clip_model.visual.output_dim
        self.points = []
        self.covs = []
        self.features = []
        self.image_ids = []
        self.max_points = []

        for b in range(self.batch_size):
            self.points.append(torch.zeros(0, 3).to(self.device))
            self.covs.append(torch.zeros(0, 3, 3).to(self.device))
            self.features.append(torch.zeros(0, self.nc).to(self.device))
            self.image_ids.append(torch.zeros(0, 1).to(self.device))
            self.max_points.append(0)

        grid_x = torch.Tensor(range(self.map_size)).to(self.device).view(1, 1, -1, 1).repeat(1, 1, 1, self.map_size)
        grid_y = torch.Tensor(range(self.map_size)).to(self.device).view(1, 1, 1, -1).repeat(1, 1, self.map_size, 1)
        self.grid = torch.cat([grid_x, grid_y], dim=1).repeat(self.batch_size, 1, 1, 1)

        self.point_merge = point_merge 
        self.dist_merge_thresh = dist_merge_threshold
        self.sim_merge_thresh = sim_merge_threshold
        self.num_image = 0

    def reset(self):

        self.points = []
        self.covs = []
        self.features = []
        self.image_ids = []
        self.max_points = []
        for b in range(self.batch_size):
            self.points.append(torch.zeros(0, 3).to(self.device))
            self.covs.append(torch.zeros(0, 3, 3).to(self.device))
            self.features.append(torch.zeros(0, self.nc).to(self.device))
            self.image_ids.append(torch.zeros(0, 1).to(self.device))
            self.max_points.append(0)
        self.num_images = 0

    def add(self, inputs) -> dict:

        rgb = inputs["rgb"].float().to(self.device) / 255.0
        depth = inputs["depth"].clone().to(self.device)

        # calc points
        depth = depth.view(self.batch_size, 1, -1)
        depth_valid = torch.logical_and(depth[:, 0, :] > 0, depth[:, 0, :] < 1)
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

        width = rgb.size(3)
        height = rgb.size(2)
        points = points.reshape(self.batch_size, 4, height, width)[:, :3, :, :]

        input_size = 224
        for b in range(self.batch_size):
            # make patches
            patched_rgb = []
            patched_points = []
            patched_covs = []
            for scale in self.scales:
                patch_size = int(input_size * (2**scale))
                for w in range(width//patch_size):
                    woffset = (width - width // patch_size * patch_size) // 2

                    for h in range(height//patch_size):
                        hoffset = (height - height // patch_size * patch_size) // 2
                        patch_rgb = rgb[b, :, hoffset+h*patch_size:hoffset+(h+1)*patch_size, woffset+w*patch_size:woffset+(w+1)*patch_size].unsqueeze(0)
                        patch_points = points[b, :, hoffset+h*patch_size:hoffset+(h+1)*patch_size, woffset+w*patch_size:woffset+(w+1)*patch_size].unsqueeze(0)

                        patch_rgb = F.resize(patch_rgb, size=(input_size, input_size))
                        patch_points = patch_points.reshape(patch_points.size(0), patch_points.size(1), -1)
                        patch_mean = patch_points.mean(dim=-1)
                        centered_points = patch_points - patch_mean.unsqueeze(2)
                        cov_matrix = torch.matmul(centered_points, centered_points.transpose(1, 2)) / (patch_points.size(2) - 1)
                        patched_rgb.append(patch_rgb)
                        patched_points.append(patch_mean)
                        patched_covs.append(cov_matrix)

            patched_rgb = torch.cat(patched_rgb, dim=0)
            patched_points = torch.cat(patched_points, dim=0)
            patched_covs = torch.cat(patched_covs, dim=0)
            patched_image_ids = torch.ones(patched_points.size(0), 1).to(torch.int).to(self.device) * self.num_images
            self.num_images += 1

            # Calc CLIP
            patched_clip = self.clip_model.encode_image(patched_rgb)
            normed_clip = patched_clip / (patched_clip.norm(dim=1, keepdim=True) + 1e-1)

            # concate points and features
            self.features[b] = torch.cat([self.features[b], normed_clip], dim=0)
            self.points[b] = torch.cat([self.points[b], patched_points], dim=0)
            self.covs[b] = torch.cat([self.covs[b], patched_covs], dim=0)
            self.image_ids[b] = torch.cat([self.image_ids[b], patched_image_ids], dim=0)

            # point merge
            self.max_points[b] += patched_points.size(0)
            if self.point_merge:

                diff = self.points[b].unsqueeze(1) - self.points[b].unsqueeze(0)
                cov_avg = (self.covs[b].unsqueeze(1) + self.covs[b].unsqueeze(0)) / 2
                cov_avg = torch.eye(3).to(self.device).unsqueeze(0).unsqueeze(0) * 1e-6
                cov_inv = torch.inverse(cov_avg)
                distances = torch.einsum('...i,...ij,...j->...', diff, cov_inv, diff)
                distances = torch.sqrt(distances) # [n, n]

                det = torch.det(self.covs[b]) # [n]
                diff_det = det.unsqueeze(1) - det.unsqueeze(0) # [n, n]
                sim_clip = self.features[b] @ self.features[b].t() # [n, n]
                keep_mask = (distances > self.dist_merge_thresh) | (diff_det >= 0) | (sim_clip < self.sim_merge_thersh) # [n, n]
                keep_mask = keep_mask.all(dim=1) # [n]

                # concate points and features
                self.features[b] = self.features[b][keep_mask]
                self.points[b] = self.points[b][keep_mask]
                self.covs[b] = self.covs[b][keep_mask]
                self.image_ids[b] = self.image_ids[b][keep_mask]

            #print("num_points = " + str(self.points[b].size(0)) + "/" + str(self.max_points[b]))

        outputs = {
            "feature_points": {
                "points": self.points,
                "covariances": self.covs,
                "features": self.features,
            },
        }

        return outputs

    def find_topk(self, inputs, k=1, image_dir=None):
                
        batch_image_ids = []
        batch_goals = []
        batch_goals2d = []
        for b in range(self.batch_size):
        
            if type(inputs) == str:
                prompt = inputs
            elif type(inputs) == dict:
                prompt = inputs["objectgoal"][b]
            elif type(inputs) == list:
                prompt = inputs[b]

            text_inputs = torch.cat([self.tokenizer("a " + prompt)]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= (text_features.norm(dim=-1, keepdim=True) + 1e-7)
            text_features = text_features.float()
            text_features = text_features.unsqueeze(-1)

            similarity = (self.features[b] @ text_features).squeeze()

            topk_sims, topk_ids = torch.topk(similarity, k)            

            goals = self.points[b][topk_ids]
            
            if goals.size(0) == 0:
                goals = None
                goals2d = None
            else:
                goals2d = goals[:, [2, 0]]
                goals2d[:, 0] = - goals2d[:, 0]
            
            batch_goals.append(goals)
            batch_goals2d.append(goals2d)

            # image ids
            image_ids = self.image_ids[b][topk_ids]
            batch_image_ids.append(image_ids)
            
        outputs = {
            "goal_xyz": batch_goals,
            "goal2d_xy": batch_goals2d,
        }

        if image_dir is not None:

            images = []
            for image_ids in batch_image_ids:
                for img_idx in image_ids:
                    img_idx = int(img_idx.item())
                    img_name = f"{img_idx:06}.png"
                    img_path = image_dir + "/" + img_name

                    image = transforms.ToTensor()(Image.open(img_path)).to(self.device)
                    image = 255 * image
                    image = image.to(torch.int)
                    
                    images.append(image)              

            if len(images) > 0:
                outputs["rgb"] = torch.stack(images)

        return outputs


    def find(self, inputs, similarity_thresh=0.29, k=0, image_dir=None):

        if k > 0:
            return self.find_topk(inputs, k, image_dir)

        batch_image_ids = []
        batch_goals = []
        batch_goals2d = []
        for b in range(self.batch_size):
        
            if type(inputs) == str:
                prompt = inputs
            elif type(inputs) == dict:
                prompt = inputs["objectgoal"][b]
            elif type(inputs) == list:
                prompt = inputs[b]

            text_inputs = torch.cat([self.tokenizer("a " + prompt)]).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= (text_features.norm(dim=-1, keepdim=True) + 1e-7)
            text_features = text_features.float()
            text_features = text_features.unsqueeze(-1)

            similarity = (self.features[b] @ text_features).squeeze()
            
            goals = self.points[b][similarity > similarity_thresh]
            
            if goals.size(0) == 0:
                goals = None
                goals2d = None
            else:
                goals2d = goals[:, [2, 0]]
                goals2d[:, 0] = - goals2d[:, 0]
            
            batch_goals.append(goals)
            batch_goals2d.append(goals2d)

            # image ids
            image_ids = self.image_ids[b][similarity > similarity_thresh]
            batch_image_ids.append(image_ids)
            
        outputs = {
            "goal_xyz": batch_goals,
            "goal2d_xy": batch_goals2d,
        }

        if image_dir is not None:

            images = []
            for image_ids in batch_image_ids:
                for img_idx in image_ids:
                    img_idx = int(img_idx.item())
                    img_name = f"{img_idx:06}.png"
                    img_path = image_dir + "/" + img_name

                    image = transforms.ToTensor()(Image.open(img_path)).to(self.device)
                    image = 255 * image
                    image = image.to(torch.int)
                    
                    images.append(image)              

            if len(images) > 0:
                outputs["rgb"] = torch.stack(images)
            

        return outputs

    def save(self, filename):
        
        clip_points = {
            "points": self.points,
            "covariances": self.covs,
            "features": self.features,
            "image_ids": self.image_ids,
        }
        torch.save(clip_points, filename)

    def load(self, filename):

        clip_points = torch.load(filename)
        self.points = clip_points["points"]
        self.covs = clip_points["covariances"]
        self.features = clip_points["features"]
        self.image_ids = clip_points["image_ids"]
        for b in range(self.batch_size):
            self.points[b] = self.points[b].to(self.device)
            self.covs[b] = self.covs[b].to(self.device)
            self.features[b] = self.features[b].to(self.device)
            self.image_ids[b] = self.image_ids[b].to(self.device)

