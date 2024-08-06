#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

from chimera.simulator import Simulator
import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict
import random
import numpy as np
import torch
import pytorch3d.transforms as p3dt
import os

class MetaWorld(Simulator):
    def __init__(self, config=None, device=0, task=None, dataset="ML10", split="train", batch_size=1, 
        camera="corner",
        min_depth=0.5,
        max_depth=5.0,
        goal_visible=True,
        dt=None,
        **kwargs):

        if "MUJOCO_GL" not in os.environ["MUJOCO_GL"]:
            os.environ["MUJOCO_GL"]='egl'

        seed = 0 if not "seed" in kwargs.keys() else kwargs["seed"]
        self.num_episodes_per_task = 20 if not "num_episodes_per_task" in kwargs.keys() else kwargs["num_episodes_per_task"]
        
        self.device = device
        self.batch_size = batch_size
        if dataset == "ML1" or dataset == "MT1":
            self.task = random.choice(_env_dict.ALL_V2_ENVIRONMENTS.keys()) if task is None else task
        else:
            self.task = dataset

        if dataset == "ML1":
            self.dataset = metaworld.ML1(self.task, seed=seed)
        elif dataset == "MT1":
            self.dataset = metaworld.MT1(self.task, seed=seed)
        elif dataset == "ML10":
            self.dataset = metaworld.ML10(seed=seed)
        elif dataset == "ML45":
            self.dataset = metaworld.ML45(seed=seed)
        elif dataset == "MT10":
            self.dataset = metaworld.MT10(seed=seed)
        elif dataset == "MT50":
            self.dataset = metaworld.MT50(seed=seed)

        self.camera = camera

        self.envs = [] 
        if split == "train":
            if dataset == "ML1" or dataset == "MT1":
                for i in range(self.num_episodes_per_task):
                    env = self.dataset.train_classes[self.task](camera_name=self.camera)
                    tk = random.choice(self.dataset.train_tasks)
                    env.set_task(tk)
                    self.envs.append(env)
            else:
                for name, env_cls in self.dataset.train_classes.items():
                    for i in range(self.num_episodes_per_task):
                        env = env_cls(camera_name=self.camera)
                        tk = random.choice([tk for tk in self.dataset.train_tasks if tk.env_name == name])
                        env.set_task(tk)
                        self.envs.append(env)
        elif split == "test":
            if dataset == "ML1" or dataset == "MT1":
                for i in range(self.num_episodes_per_task):
                    env = self.dataset.test_classes[self.task](camera_name=self.camera)
                    tk = random.choice(self.dataset.test_tasks)
                    env.set_task(tk)
                    self.envs.append(env)
            for name, env_cls in self.dataset.test_classes.items():
                for i in range(self.num_episodes_per_task):
                    env = env_cls(camera_name=self.camera)
                    tk = random.choice([tk for tk in self.dataset.test_classes if tk.env_name == name])
                    env.set_task(tk)
                    self.envs.append(env)

        self.current_env_id = None

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.goal_visible = goal_visible
        self.param_dt = dt
        self.dt = None

    def get_config(self) -> dict:
        dict_conf = {
            "agent": {
                "action_type": "continuous",
            },
            "rgb": {
                "width": 480,
                "height": 480,
            },
            "depth": {
                "width": 480,
                "height": 480,
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
            }
        }
        if self.param_dt is not None and self.dt is not None:
            dict_conf["sampling_interval"] = self.dt
        return dict_conf

    def num_episodes(self) -> int:
        return len(self.envs)

    def step(self, inputs) -> (dict, dict):
        a = torch.cat([inputs["cmd_ee_pos"][0], inputs["cmd_gripper"][0]], dim=0).cpu().numpy() 
        res = self.env.step(a)
        if len(res) == 4:
            obs, reward, done, info = res
        else:
            obs, reward, _, done, info = res
        if done or self.env.curr_path_length  >= self.env.max_path_length:
            self._is_episode_over = True
        obs, info = self.refine_format(obs, reward, done, info)
        return obs, info

    def reset(self) -> (dict, dict):
        if self.current_env_id is not None:
            self.current_env_id += 1
        else:
            self.current_env_id = 0
        self.env = self.envs[self.current_env_id]
        if self.param_dt is not None:
            self.frame_skip = self.param_dt // self.env.model.opt.timestep
            self.env.frame_skip = int(self.frame_skip)
        self.dt = self.env.model.opt.timestep * self.env.frame_skip
        self._is_episode_over = False
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs, info = obs

        if not self.goal_visible:
            goalid = self.env.sim.model._site_name2id["goal"]
            self.env.sim.model.site_rgba[goalid][3] = 0

        a = self.env.action_space.sample() 
        reward, info = self.env.evaluate_state(obs, a)
        done = False
        obs, info = self.refine_format(obs, reward, done, info)
        return obs, info

    def is_episode_over(self) -> bool:
        return self._is_episode_over

    def refine_format(self, obs, reward, done, info) -> (dict, dict):
        refine_obs = {
            "ee_pos": torch.Tensor(obs[:3]).to(self.device).unsqueeze(0),
            "gripper": torch.Tensor(obs[3:4]).to(self.device).unsqueeze(0),
            "obj1_pos": torch.Tensor(obs[4:7]).to(self.device).unsqueeze(0),
            "obj1_rot": torch.Tensor(obs[7:11]).to(self.device).unsqueeze(0)[:, [3, 0, 1, 2]],
            "obj2_pos": torch.Tensor(obs[11:14]).to(self.device).unsqueeze(0),
            "obj2_rot": torch.Tensor(obs[14:18]).to(self.device).unsqueeze(0)[:, [3, 0, 1, 2]],
            "prev_ee_pos": torch.Tensor(obs[18:21]).to(self.device).unsqueeze(0),
            "prev_gripper": torch.Tensor(obs[21:22]).to(self.device).unsqueeze(0),
            "prev_obj1_pos": torch.Tensor(obs[22:25]).to(self.device).unsqueeze(0),
            "prev_obj1_rot": torch.Tensor(obs[25:29]).to(self.device).unsqueeze(0)[:, [3, 0, 1, 2]],
            "prev_obj2_pos": torch.Tensor(obs[29:32]).to(self.device).unsqueeze(0),
            "prev_obj2_rot": torch.Tensor(obs[32:36]).to(self.device).unsqueeze(0)[:, [3, 0, 1, 2]],
            "goal_pos": torch.Tensor(obs[36:]).to(self.device).unsqueeze(0),
            "raw": torch.Tensor(obs).to(self.device).unsqueeze(0)
        }
        refine_info = {
            "success": torch.Tensor([info["success"]]).to(self.device).unsqueeze(0),
            "near_object": torch.Tensor([info["near_object"]]).to(self.device).unsqueeze(0),
            "grasp_success": torch.Tensor([info["grasp_success"]]).to(self.device).unsqueeze(0),
            "grasp_reward": torch.Tensor([info["grasp_reward"]]).to(self.device).unsqueeze(0),
            "in_place_reward": torch.Tensor([info["in_place_reward"]]).to(self.device).unsqueeze(0),
            "obj_to_target": torch.Tensor([info["obj_to_target"]]).to(self.device).unsqueeze(0),
            "unscaled_reward": torch.Tensor([info["unscaled_reward"]]).to(self.device).unsqueeze(0),
            "done": torch.Tensor([done]).to(self.device).unsqueeze(0),
        }
        self.env.model.vis.map.znear = self.min_depth
        self.env.model.vis.map.zfar = self.max_depth
        self.env.render_mode = "rgb_array"
        rgb = self.env.render()
        self.env.render_mode = "depth_array"
        depth = self.env.render()
        refine_obs["rgb"] = torch.from_numpy(rgb[::-1]).permute(2, 0, 1).to(self.device).unsqueeze(0)
        refine_obs["depth"] = torch.from_numpy(depth[::-1]).to(self.device).unsqueeze(0).unsqueeze(0)
        
        return refine_obs, refine_info
