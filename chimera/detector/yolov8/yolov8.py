#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

from chimera.detector import Detector
from ultralytics import YOLO
import os
import torch
import torchvision.transforms as transforms

class YOLOv8(Detector):
    def __init__(self, config=None, device=0, batch_size=1, model = "yolov8x-seg.pt", **kwargs):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.device = device
        self.batch_size = batch_size
        self.model = YOLO(os.path.join(dir_path, model))
        self.model.to(self.device)
        self.conf_thresh = 0.8
        if config is not None:
            if "objects" in config.keys():
                self.conf_thresh = config["objects"]["conf_thresh"]

    def get_config(self):
        config = {
            "objects": {
                "names": self.model.names,
                "conf_thresh": self.conf_thresh,
            }
        }
        return config

    def reset(self):
        pass

    def detect(self, inputs):
        rgb = inputs["rgb"].clone()
        offset = [0, 0]
        height = rgb.size(2)
        width = rgb.size(3)
        if not rgb.size(2) % 32 == 0 or not rgb.size(3) % 32 == 0:
            height = rgb.size(2) // 32 * 32
            width = rgb.size(3) // 32 * 32
            offset[0] = (rgb.size(2) - height) // 2
            offset[1] = (rgb.size(3) - width) // 2
            rgb = rgb[:, :, offset[0]:offset[0]+height, offset[1]:offset[1]+width]
        results = self.model(source=rgb / 255.0, verbose=False)
        max_det = max([results[b].boxes.shape[0] for b in range(len(results))])
        boxes = -1 * torch.ones(self.batch_size, max_det, 6).to(self.device)
        masks = torch.zeros(self.batch_size, max_det, inputs["rgb"].shape[2], inputs["rgb"].shape[3]).to(self.device)
        for b in range(self.batch_size):
            if results[b].boxes.shape[0] > 0:
                boxes[b, :results[b].boxes.shape[0], :] = results[b].boxes.data
                if results[b].masks is not None:
                    masks[b, :results[b].masks.shape[0], offset[0]:offset[0]+height, offset[1]:offset[1]+width] = results[b].masks.data
        
        # apply offset to boxed
        boxes[..., 0] += offset[0]
        boxes[..., 1] += offset[1]
        boxes[..., 2] += offset[0]
        boxes[..., 3] += offset[1]
        
        outputs = {
            "objects": {
                "boxes": boxes,
                "masks": masks,
            },
        }
        return outputs
