#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import os
import sys
import argparse
import math
from chimera.localizer import Localizer
droid_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DROID-SLAM")
python_dir = os.path.join(droid_dir, "droid_slam")
sys.path.append(python_dir)
import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2
from droid import Droid
from lietorch import SE3
import pytorch3d.transforms as p3dt

def image_stream(dataset="tartan"):

    if dataset == "tartan":
        image_dir = os.path.join(droid_dir, "data", "abandonedfactory")
        calib = os.path.join(droid_dir, "calib", "tartan.txt")
        stride = 2
    elif dataset == "eth":
        image_dir = os.path.join(droid_dir, "data", "sfm_bench", "rgb")
        calib = os.path.join(droid_dir, "calib", "eth.txt")
        stride = 3
    elif dataset == "barn":
        image_dir = os.path.join(droid_dir, "data", "Barn")
        calib = os.path.join(droid_dir, "calib", "barn.txt")
        stride = 1
    elif dataset == "euroc":
        image_dir = os.path.join(droid_dir, "data", "mav0", "cam0", "data")
        calib = os.path.join(droid_dir, "calib", "euroc.txt")
        stride = 3
    elif dataset == "tum3":
        image_dir = os.path.join(droid_dir, "data", "rgbd_dataset_freiburg3_cabinet", "rgb")
        calib = os.path.join(droid_dir, "calib", "tum3.txt")
        stride = 1
    else:
        raise NotImplementedError("There are no dataset name:" + dataset)

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    image_list = sorted(os.listdir(image_dir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(image_dir, imfile))
        if (len(calib) > 4):
            image = cv2.undistort(image, K, calib[4:])

        image = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0)

        intrinsics = [fx, fy, cx, cy]

        yield t, image, intrinsics
        
class DroidSLAM(Localizer):
    def __init__(self, config, device=0, batch_size=1, 
        buffer = 512,
        beta = 0.3,
        filter_thresh = 2.4,
        warmup = 8,
        keyframe_thresh = 4.0,
        frontend_thresh = 16.0,
        frontend_window = 25,
        frontend_radius = 2,
        frontend_nms = 1,
        backend_thresh = 22.0,
        backend_radius = 2,
        backend_nms = 3,
        upsample = False,
        stereo = False,
        **kwargs):

        self.config = config
        self.device = device

        # image size
        w0 = config["rgb"]["width"]
        h0 = config["rgb"]["height"]
        self.resize_width = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        self.resize_height = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        self.image_width = self.resize_width - self.resize_width % 8
        self.image_height = self.resize_height - self.resize_height % 8
        
        # intrinsics    
        if "intrinsics" in config["rgb"].keys():
            fx = config["rgb"]["intrinsics"][0]
            fy = config["rgb"]["intrinsics"][4]
            cx = config["rgb"]["intrinsics"][2]
            cy = config["rgb"]["intrinsics"][5]
            self.intrinsics = torch.as_tensor([fx, fy, cx, cy])
        else:      
            hfov = config["rgb"]["hfov"]
            f = 0.5 * w0 / math.tan(0.5 * hfov / 180.0 * math.pi)
            cx = 0.5 * w0
            cy = 0.5 * h0
            self.intrinsics = torch.as_tensor([f, f, cx, cy])
        self.intrinsics[0::2] *= (self.resize_width / w0)
        self.intrinsics[1::2] *= (self.resize_height / h0)

        if "depth" in config.keys():
            self.depth_width = config["depth"]["width"]
            self.depth_height = config["depth"]["height"]
            self.min_depth = config["depth"]["min_depth"]
            self.max_depth = config["depth"]["max_depth"]
            self.depth_hfov = config["depth"]["hfov"]

        args = argparse.Namespace()
        args.weights = os.path.join(droid_dir, "droid.pth")
        args.buffer = buffer
        args.image_size = [self.image_height, self.image_width]
        args.disable_vis = True
        args.beta = beta
        args.filter_thresh = filter_thresh
        args.warmup = warmup
        args.keyframe_thresh = keyframe_thresh
        args.frontend_thresh = frontend_thresh
        args.frontend_window = frontend_window
        args.frontend_radius = frontend_radius
        args.frontend_nms = frontend_nms
        args.backend_thresh = backend_thresh
        args.backend_radius = backend_radius
        args.backend_nms = backend_nms
        args.upsample = upsample
        args.stereo = stereo
        self.args = args

        # Change the cuda device
        prev_device = torch.cuda.current_device()
        torch.cuda.set_device(self.device)
        
        self.droid = Droid(self.args)

        torch.cuda.set_device(prev_device)        

        self.R = torch.diag(torch.Tensor([1, -1, -1, 1]).to(self.device))
        self.image_stream = []

    def reset(self):
        # Change the cuda device
        prev_device = torch.cuda.current_device()
        torch.cuda.set_device(self.device)

        self.droid = Droid(self.args)
        self.image_stream = []

        torch.cuda.set_device(prev_device)   

    def is_warmuped(self):
        return self.droid.video.counter.value > self.args.warmup

    def track(self, inputs):

        t = inputs["timestamp"][0]
        rgb = inputs["rgb"][0].unsqueeze(0)
        rgb = F.resize(rgb, size=(self.resize_height, self.resize_width), antialias=None)
        rgb = rgb[:, :, :self.image_height, :self.image_width]
        depth = None        
        if "depth" in inputs.keys():
            depth = inputs["depth"][0]
            depth = F.resize(depth, size=(self.resize_height, self.resize_width), antialias=None)
            depth = depth.squeeze()
            depth = self.min_depth + depth * (self.max_depth - self.min_depth)
            depth = depth[:self.image_height, :self.image_width]

        # Change the cuda device
        prev_device = torch.cuda.current_device()
        torch.cuda.set_device(self.device)

        self.droid.track(t, image=rgb, depth=depth, intrinsics=self.intrinsics)
    
        c = self.droid.video.counter.value
        traj_mat = SE3(self.droid.video.poses[:c]).inv().matrix()
        traj_rot = self.R.inverse() @ traj_mat
        traj = torch.zeros(traj_rot.shape[0], 7).to(self.device)
        traj[:, :3] = traj_rot[:, :3, 3]
        traj[:, 3:] = p3dt.matrix_to_quaternion(traj_mat[:, :3, :3])
        traj[:, 5] = -traj[:, 5]

        torch.cuda.set_device(prev_device)   

        outputs = {
            "trajectory": traj.unsqueeze(0),
            "position": traj[-1][:3].unsqueeze(0),
            "rotation": traj[-1][3:].unsqueeze(0),
        }                

        return outputs

