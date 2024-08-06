#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

from chimera.detector import Detector
from segment_anything import SamPredictor, sam_model_registry
import torch
import torchvision.transforms.functional as F
import os

class Sam(Detector):
    def __init__(self, config=None, device=0, batch_size=1, model_type = "vit_h", model="sam_vit_h_4b8939.pth", **kwargs):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.device = device
        self.batch_size = batch_size
        self.model = os.path.join(dir_path, model)
        self.sam = sam_model_registry[model_type](checkpoint=self.model).to(self.device)
        self.predictor = SamPredictor(self.sam)

    def reset(self):
        pass

    def detect(self, inputs, **kwargs):
        outputs = {}
        outputs["segments"] = {}
        for b in range(self.batch_size):
            prompts = inputs["objectgoal"][b]
            rgb = inputs["rgb"][b].unsqueeze(0)
            original_size = (rgb.shape[2], rgb.shape[3])
            rgb = F.resize(rgb, size=(round(1024 * rgb.shape[2]/rgb.shape[3]), 1024), antialias=None)
            self.predictor.set_torch_image(rgb, original_size)
            masks, _, _ = self.predictor.predict_torch(None, prompts, None, None, True, True)
            outputs["segments"][prompts] = masks
        return outputs
