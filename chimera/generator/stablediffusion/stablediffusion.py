#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from chimera.generator import Generator

class StableDiffusion(Generator):
    def __init__(self, config=None, device=0, model="stabilityai/stable-diffusion-2-1", mode="txt2img", **kwargs):
        if mode == "txt2img":
            self.pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
        elif mode == "img2img":
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=torch.float16)

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.device = device
        self.pipe = self.pipe.to(self.device)

    def __call__(self, prompt, **kwargs):
        images = self.pipe(prompt, **kwargs).images

        return images

