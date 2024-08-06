#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import copy
import math
import numpy as np
import torch
import pytorch3d.transforms as p3dt

def deepcopy_dict_with_tensors(d):
    """
    Recursive function to deep copy a dict with tensors.
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = deepcopy_dict_with_tensors(v)
        elif isinstance(v, torch.Tensor):
            new_dict[k] = v.clone()
        else:
            new_dict[k] = v
    return new_dict

def estimate_dpose(position, rotation, prev_position, prev_rotation):
    direction_vector = position - prev_position
    inv_prot = p3dt.quaternion_invert(prev_rotation)
    dpos = p3dt.quaternion_apply(inv_prot, direction_vector)
    drot = p3dt.quaternion_multiply(inv_prot, rotation)
    return dpos, drot

def transform_position_to_2d(position):
    position2d = position[:, [2, 0]]
    position2d[:, 0] = -position2d[:, 0]
    return position2d

def transform_rotation_to_2d(rotation):
    direction_vector = torch.Tensor([0, 0, -1]).to(rotation.device).repeat(rotation.shape[0], 1)
    heading_vector = p3dt.quaternion_apply(rotation, direction_vector)
    phi = -torch.arctan2(heading_vector[...,0], -heading_vector[...,2])
    return phi

def transform_goal2d_to_xy(goal2d, position2d, rotation2d):
    r = rotation2d + goal2d[:, 1:2]
    goal2d_xy = position2d + goal2d[:, 0] * torch.cat([torch.cos(r), -torch.sin(r)], dim=1)
    return goal2d_xy

def calc_local_points(config, depth):
    device = depth.device
    batch_size = depth.shape[0]
    
    # parameters about depth
    depth_width = config["depth"]["width"]
    depth_height = config["depth"]["height"]
    min_depth = config["depth"]["min_depth"]
    max_depth = config["depth"]["max_depth"]
    depth_hfov = config["depth"]["hfov"]

    # camera intrinsics
    f = 0.5 * depth_width / math.tan(0.5 * depth_hfov / 180.0 * math.pi)
    cx = 0.5 * depth_width
    cy = 0.5 * depth_height
    K = np.array([[f, 0.0, cx, 0.0],
        [0.0, f, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]], dtype="float32")
    inv_K = np.linalg.pinv(K)
    K = torch.from_numpy(K).to(device)
    inv_K = torch.from_numpy(inv_K).to(device)

    # pix coords
    meshgrid = np.meshgrid(range(depth_width), range(depth_height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).to(device)
    pix_coords = torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
    pix_coords = pix_coords + 0.5
    ones = torch.ones(1, depth_height * depth_width).to(device)
    pix_coords = torch.cat([pix_coords, ones], 0)        
    camera_coords = inv_K[:3, :3] @ pix_coords

    depth = depth.view(batch_size, 1, -1)
    depth_valid = torch.logical_and(depth[:, 0, :,] > 0, depth[:, 0, :] < 1)
    scaled_depth = min_depth + depth * (max_depth - min_depth)
    local_points = scaled_depth * camera_coords
    local_points = local_points.view(batch_size, 3, depth_height, depth_width)

    return local_points    

def calc_reachable_goal2d_xy(config, goal2d, position2d, rotation2d, map2d):
    r = rotation2d + goal2d[:, 1:2]
    map_pos = torch.round(position2d / config["map2d"]["scale"] + config["map2d"]["size"])
    map_pos = map_pos.to(torch.int64)
    map_range = torch.ceil(goal2d[:, 0] / config["map2d"]["scale"])
    map_range = map_range.to(torch.int64)
    map_goal = map_pos.clone()
    for b in range(position2d.shape[0]):
        for l in range(map_range[b].item()):
            cgoal = map_pos[b] + torch.round(l[b] * torch.cat([torch.cos(r[b]), -torch.sin(r[b])], dim=1)).to(torch.int64) 
            if map2d[b, 0, cgoal[b, 1], cgoal[b, 0]].item() < 1:
                map_goal[b] = cgoal
            else:
                break
    goal2d_xy = (map_goal - config["map2d"]["size"] / 2) * config["map2d"]["scale"] 
    return goal2d_xy

def box_to_goal2d(config, depth, box):
    local_points = calc_local_points(config, depth)

    points = torch.zeros(box.shape[0], 3).to(box.device)
    for b in range(box.shape[0]):
        box_points = local_points[b, :, int(box[b, 0, 1].item()):int(box[b, 0, 3].item()), int(box[b, 0, 0].item()):int(box[b, 0, 2].item())]
        box_points = box_points.reshape(3, -1)
        points[b] = torch.mean(box_points, dim=1)
    range2d = torch.norm(points[:, [0, 2]], dim=1).unsqueeze(1)
    phi = -torch.atan2(points[:, 0],  points[:, 2]).unsqueeze(1)
    goal2d = torch.cat([range2d, phi], dim=1)
    return goal2d

def mask_to_goal2d(config, depth, mask):
    local_points = calc_local_points(config, depth)

    points = torch.zeros(mask.shape[0], 3).to(mask.device)
    for b in range(box.shape[0]):
        mask_points = local_points[b, :, mask[b, :]]
        points[b] = torch.mean(mask_points.view(3, -1), dim=1)
    range2d = torch.norm(points[:, [0, 2]], dim=1).unsqueeze(1)
    phi = -torch.atan2(points[:, 0], -points[:, 2]).unsqueeze(1)
    goal2d = torch.cat([range2d, phi], dim=1)
    return goal2d

def check_object_category(name1, name2):
    if name1 == name2:
        return True
    if (name1 == "couch" and name2 == "sofa") or (name1 == "sofa" and name2 == "couch"):
        return True
    if (name1 == "plant" and name2 == "potted plant") or (name1 == "potted plant" and name2 == "plant"):
        return True
    if (name1 == "tv" and name2 == "tv_monitor") or (name1 == "tv_monitor" and name2 == "tv"):
        return True
    return False

def detect_collision(config, position2d, prev_position2d, prev_action=None, prev_cmd_vel=None, thresh=0.01):
    collision = torch.zeros(position2d.shape[0], 1).to(position2d.device)
    if config["agent"]["action_type"] == "discrete":
        forward = prev_action == 1 # Move Forward
        dist = torch.norm(position2d - prev_position2d, dim=1)
        stop = dist < config["agent"]["forward_step_size"] * thresh
        collision[forward and stop] = 1
    elif config["agent"]["action_type"] == "continuous":
        forward = (prev_cmd_vel[:, :1] > 0.1)
        dist = torch.norm(position2d - prev_position2d, dim=1)
        stop = dist < prev_cmd_vel[:, 0] * config["sampling_interval"] * thresh
        collision[forward and stop] = 1
    return collision

def expand_inputs(inputs):
    outputs = deepcopy_dict_with_tensors(inputs)
    if "position2d" not in inputs.keys():
        if "position" in inputs.keys():
            outputs["position2d"] = transform_position_to_2d(inputs["position"])
    if "rotation2d" not in inputs.keys():
        if "rotation" in inputs.keys():
            outputs["rotation2d"] = transform_rotation_to_2d(inputs["rotation"])
    if "dpos2d" not in inputs.keys():
        if "dpos" in inputs.keys():
            outputs["dpos2d"] = transform_position_to_2d(inputs["dpos"])
    if "drot2d" not in inputs.keys():
        if "drot" in inputs.keys():
            outputs["drot2d"] = transform_rotation_to_2d(inputs["drot"])
    if "goal2d_xy" not in inputs.keys():
        if "goal2d" in inputs.keys():
            if "position2d" in outputs.keys() and "rotation2d" in outputs.keys():
                outputs["goal2d_xy"] = transform_goal2d_to_xy(inputs["goal2d"], outputs["position2d"], outputs["rotation2d"])
    return outputs


