#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image
import subprocess

class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

class Visualizer:
    def __init__(self, config, name="visualize", device=0, show=True, wait=1, height=360, ego=True, ego_size=180, hide_ego=False, show_traj=True,  trans_heatmap=0.5, show_rgbmap=False, traj_color=[255, 0, 0]):
        self.config = config
        self.name = name
        self.device = device
        if "objects" in self.config.keys():
            if "conf_thresh" in self.config["objects"].keys():
                self.conf_thresh = self.config["objects"]["conf_thresh"]
        self.show = show
        self.wait = wait
        self.height = height;
        self.ego = ego
        self.ego_size = ego_size
        self.hide_ego = hide_ego
        self.show_traj = show_traj
        self.show_rgbmap = show_rgbmap
        self.text_size = self.height / 960.0;
        self.text_thickness = math.ceil(self.height / 360.0);
        self.trans_heatmap = trans_heatmap
        self.traj_color = traj_color

        if "objects" in self.config.keys():
            if "conf_thresh" in self.config["objects"].keys():
                self.conf_thresh = self.config["objects"]["conf_thresh"]
        # logs
        self.images = []
        self.trajectory = []

    def reset(self):

        self.images = []
        self.trajectory = []

    def visualize(self, inputs, **kwargs):

        images = []

        if "rgb" in inputs.keys():
            rgb = inputs["rgb"].clone()
            rgb = F.resize(rgb, size=(self.height, round(self.height * rgb.shape[3] / rgb.shape[2])), antialias=None)

            # Draw object detection results
            if "objects" in inputs.keys():
                object_boxes = inputs["objects"]["boxes"].clone()
                scale = self.height / self.config["rgb"]["height"]
                for b in range(rgb.shape[0]):
                    im = cv2.cvtColor(np.uint8(rgb[b].permute(1, 2, 0).cpu().numpy()), cv2.COLOR_RGB2BGR)
                    for n in range(object_boxes.shape[1]):
                        conf = object_boxes[b, n, 4].item()
                        if conf < self.conf_thresh:
                            continue
                        
                        pt1 = (int(object_boxes[b, n, 0].item() * scale), int(object_boxes[b, n, 1].item() * scale))
                        pt2 = (int(object_boxes[b, n, 2].item() * scale), int(object_boxes[b, n, 3].item() * scale))
                        ptt = (int(object_boxes[b, n, 0].item() * scale), int(object_boxes[b, n, 1].item() * scale + self.text_thickness))
                        cls = int(object_boxes[b, n, 5].item())
                        color = colors(cls, bgr=True)
                        cv2.rectangle(im, pt1=pt1, pt2=pt2, color=color, thickness=3)
                        cv2.putText(im, self.config["objects"]["names"][cls], pt1, cv2.FONT_HERSHEY_SIMPLEX, self.text_size * 2, color, self.text_thickness, cv2.LINE_AA) 
                    rgb[b] = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).to(inputs["rgb"].device).permute(2, 0, 1)

            images.append(rgb)

        if "depth" in inputs.keys():
            depth = inputs["depth"].clone()
            depth = F.resize(depth, size=(self.height, round(self.height * depth.shape[3] / depth.shape[2])), antialias=None)
            depth = depth * 255.0
            depth = depth.to(torch.int32)
            depth = torch.cat([depth for _ in range(3)], dim=1)
            images.append(depth)

        if "map2d" in inputs.keys():
            map2d = inputs["map2d"].clone()
            map2d = map2d * 255.0
            batch_size = map2d.shape[0]
            map_scale = self.config["map2d"]["scale"]
            map_size = self.config["map2d"]["size"]
            map2d = torch.cat([map2d for _ in range(3)], dim=1)
            device = map2d.device

            if "rgbmap2d" in inputs.keys() and self.show_rgbmap:
                map2d = inputs["rgbmap2d"].clone()

            if "semmap2d" in inputs.keys():
                map2d = map2d.permute(0, 2, 3, 1)
                semmap2d = inputs["semmap2d"].clone()
                for cls in range(semmap2d.shape[1]): 
                    color =  torch.Tensor(colors(cls, bgr=False)).to(device)
                    for b in range(semmap2d.shape[0]):
                        map2d[b, semmap2d[b, cls] > self.conf_thresh] = color
                map2d = map2d.permute(0, 3, 1, 2)

            if "heatmap2d" in inputs.keys():
                heatmap2d = inputs["heatmap2d"]
                heatmap2d = heatmap2d.permute(0, 2, 3, 1)
                heatmap2drgb = torch.zeros(heatmap2d.shape[0], heatmap2d.shape[1], heatmap2d.shape[2], 3).to(self.device) 
                for b in range(map2d.shape[0]):
                    np_heatmap2d = (heatmap2d[b].cpu().numpy() * 255).astype(np.uint8)
                    np_heatmap2d = cv2.applyColorMap(np_heatmap2d, cv2.COLORMAP_JET)
                    np_heatmap2d = np_heatmap2d[:, :, ::-1].astype(np.float32)  # convert to RGB
                    heatmap2drgb[b] = torch.from_numpy(np_heatmap2d).to(self.device)
                heatmap2drgb = heatmap2drgb.permute(0, 3, 1, 2)
                map2d = self.trans_heatmap * heatmap2drgb + (1-self.trans_heatmap) * map2d                

            if "position2d" in inputs.keys():
                agent_radius = 0.18#self.config["agent"]["radius"]
                ar = agent_radius / map_scale
                pos = inputs["position2d"] / map_scale + map_size / 2
                grid_x = torch.Tensor(range(map_size)).to(device).view(1, 1, -1, 1).repeat(1, 1, 1, map_size)
                grid_y = torch.Tensor(range(map_size)).to(device).view(1, 1, 1, -1).repeat(1, 1, map_size, 1)
                grid = torch.cat([grid_x, grid_y], dim=1).repeat(batch_size, 1, 1, 1)
                pos = pos[:, [1, 0]]
                if "path" in inputs.keys():
                    path = inputs["path"].clone()
                    for b in range(map2d.shape[0]):
                        map2d[b, 0, path[b, :, 1], path[b, :, 0]] = 0
                        map2d[b, 1, path[b, :, 1], path[b, :, 0]] = 255
                        map2d[b, 2, path[b, :, 1], path[b, :, 0]] = 0
                if "landmark" in inputs.keys():
                    def draw_landmarks(landmarks):
                        landmarks = landmarks / map_scale + map_size / 2
                        landmarks = landmarks[..., [1, 0]]
                        if landmarks.dim() < 2:
                            landmarks.unsqueeze(0)
                        for n in range(landmarks.size(0)):
                            landmark = torch.norm(grid.permute(2, 3, 0, 1) - landmarks[n].unsqueeze(0), dim=3).permute(2, 0, 1) < 2 * ar
                            map2d[:, 0][landmark] = 255
                            map2d[:, 1][landmark] = 0
                            map2d[:, 2][landmark] = 0
                    landmarks = inputs["landmark"]
                    if isinstance(landmarks, list):
                        for b_landmarks in landmarks:
                            draw_landmarks(b_landmarks)
                    else:
                        draw_goals(landmarks)
                if "goal2d_xy" in inputs.keys():
                    def draw_goals(goals):
                        goals = goals / map_scale + map_size / 2
                        goals = goals[..., [1, 0]]
                        if goals.dim() < 2:
                            goals.unsqueeze(0)
                        for n in range(goals.size(0)):
                            goal = torch.norm(grid.permute(2, 3, 0, 1) - goals[n].unsqueeze(0), dim=3).permute(2, 0, 1) < 2 * ar
                            map2d[:, 0][goal] = 0
                            map2d[:, 1][goal] = 255
                            map2d[:, 2][goal] = 0
                    goals = inputs["goal2d_xy"]
                    if isinstance(goals, list):
                        for b_goals in goals:
                            draw_goals(b_goals)
                    else:
                        draw_goals(goals)
                            
                if not self.hide_ego:
                    agent = torch.norm(grid.permute(2, 3, 0, 1) - pos, dim=3).permute(2, 0, 1) < ar
                    map2d[:, 0][agent] = self.traj_color[0]
                    map2d[:, 1][agent] = self.traj_color[1]
                    map2d[:, 2][agent] = self.traj_color[2]
                    self.add_trajectory(pos)
                    if self.show_traj:
                        traj = torch.cat(self.trajectory, dim=1)
                        for b in range(map2d.shape[0]):
                            map2d[b, 0, traj[b, :, 0], traj[b, :, 1]] = self.traj_color[0]
                            map2d[b, 1, traj[b, :, 0], traj[b, :, 1]] = self.traj_color[1]
                            map2d[b, 2, traj[b, :, 0], traj[b, :, 1]] = self.traj_color[2]

                if self.ego:
                    map2d_new = torch.zeros(map2d.shape[0], 3, self.ego_size, self.ego_size).to(device)
                    for b in range(map2d.shape[0]):
                        w = self.ego_size // 2
                        pos = torch.round(pos).to(torch.int64)
                        map2d_new[b] = map2d[b, :, pos[b,0]-w:pos[b,0]+w, pos[b,1]-w:pos[b,1]+w]
                    map2d = map2d_new
                else:
                    map2d_new = torch.zeros(map2d.shape[0], 3, self.ego_size, self.ego_size).to(device)
                    for b in range(map2d.shape[0]):
                        w = self.ego_size // 2
                        origin = map2d.shape[2] // 2
                        map2d_new[b] = map2d[b, :, origin-w:origin+w, origin-w:origin+w]
                    map2d = map2d_new


            map2d = F.resize(map2d, size=(self.height, round(self.height * map2d.shape[3] / map2d.shape[2])), antialias=None)            

            # Draw instruction/question
            if "instruction" in inputs.keys() or "question" in inputs.keys():
                if "instruction" in inputs.keys():
                    text = inputs["instruction"]
                elif "question" in inputs.keys():
                    text = inputs["question"]
                for b in range(rgb.shape[0]):
                    im = cv2.cvtColor(np.uint8(map2d[b].permute(1, 2, 0).cpu().numpy()), cv2.COLOR_RGB2BGR)
                    color = (0, 255, 0)
                    words = text[b].split()
                    lines = []
                    line = ""
                    for word in words:
                        if cv2.getTextSize(line + word, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, 2)[0][0] > self.height:
                            lines.append(line.strip())
                            line = ""
                        line += word + " "
                    if line:
                        lines.append(line.strip())
                    for i, line in enumerate(lines):
                        text_pos = (0, int(25 * (i + 1) * self.text_size))
                        cv2.putText(im, line, text_pos, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, color, self.text_thickness, cv2.LINE_AA) 
                    map2d[b] = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).to(inputs["map2d"].device).permute(2, 0, 1)

            images.append(map2d)

        image = None
        if len(images) > 0:
            image = torch.cat(images, dim=3).to(torch.int32)
            image = torch.cat([image[b] for b in range(image.shape[0])], dim=1)

        if self.show and image is not None:
            img = cv2.cvtColor(np.uint8(image.permute(1, 2, 0).cpu().numpy()), cv2.COLOR_RGB2BGR)
            cv2.imshow(self.name, img)
            cv2.waitKey(self.wait)

        # save logs
        self.images.append(image)

        return image

    def add_trajectory(self, pos):

        if len(self.trajectory) == 0:
            self.trajectory.append(pos.unsqueeze(1).to(torch.int64))
            return

        last_pos = self.trajectory[-1]

        maxdiff_x = torch.max(torch.abs(pos[:, 0] - last_pos[:, 0, 0]), dim=0)[0].item()
        maxdiff_y = torch.max(torch.abs(pos[:, 1] - last_pos[:, 0, 1]), dim=0)[0].item()

        maxdiff = math.ceil(max(maxdiff_x, maxdiff_y))

        for i in range(maxdiff):
            dpos = float(i) / maxdiff * (pos[:] - last_pos[:, 0])
            p = last_pos[:, 0] + dpos
            self.trajectory.append(p.unsqueeze(1).to(torch.int64))
        return

    def save_video(self, output_dir, video_name, codec="mpeg4", fps=10, quality=5, verbose=True, **kwargs):

        channel, height, width = self.images[0].shape
        size = (width, height)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        video_name = video_name.replace(" ", "_").replace("\n", "_")

        # File names are not allowed to be over 255 characters
        video_name_split = video_name.split("/")
        video_name = "/".join(
            video_name_split[:-1] + [video_name_split[-1][:251] + ".mp4"]
        )
        output_file = os.path.join(output_dir, video_name)

        # ffmpeg
        process = subprocess.Popen([
            'ffmpeg',
            '-y', # overwrite
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', # size
            '-pix_fmt', 'bgr24',
            '-r', str(fps), # fps
            '-i', '-', # standard input
            '-an', # no audio
            '-vcodec', codec, # codec
            '-q:v', str(quality), # quality
            output_file
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        for image in self.images:
            img = cv2.cvtColor(np.uint8(image.permute(1, 2, 0).cpu().numpy()), cv2.COLOR_RGB2BGR)
            process.stdin.write(img.tobytes())
        process.stdin.close()
        process.wait()

def create_visualizer(config, name="visualize", show=True, wait=1, height=360, ego=True, ego_size=180, hide_ego=False, show_rgbmap=False):
    return Visualizer(config=config, name=name, show=show, wait=wait, height=height, ego=ego, ego_size=ego_size, hide_ego=hide_ego, show_rgbmap=show_rgbmap)

def visualize(config, inputs:dict, name="visualize", show=True, wait=1, height=360, ego=True, ego_size=180, hide_ego=False, show_rgbmap=False) -> torch.Tensor:

    vis = Visualizer(config=config, name=name, show=show, wait=wait, height=height, ego=ego, ego_size=ego_size, hide_ego=hide_ego, show_rgbmap=show_rgbmap)
    return vis.visualize(inputs=inputs)        

