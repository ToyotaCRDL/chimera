#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

from chimera.simulator import Simulator
import os
import numpy as np
import quaternion
import habitat
#from habitat.utils.visualizations import maps
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
import torch
from pathlib import Path
from omegaconf import OmegaConf

class Habitat(Simulator):
    def __init__(self, config=None, device=0, task="objectnav", dataset="hm3d", split="val_mini", batch_size=1, **kwargs):
        self.habitat_path = os.path.dirname(os.path.abspath(__file__))
        self.task = task
        self.dataset = dataset

        self.rearranges = []
        for p in Path(os.path.join(self.habitat_path, "habitat-lab/habitat-lab/habitat/config/benchmark/rearrange/")).glob("*.yaml"):
            self.rearranges.append(p.stem)
        if self.task == "pointnav" or self.task == "objectnav" or self.task == "imagenav":
            if self.dataset == "habitat-test-scenes":
                config_path = os.path.join(self.habitat_path, "habitat-lab/habitat-lab/habitat/config/benchmark/nav/", self.task, self.task + "_habitat_test.yaml")
            else:
                config_path = os.path.join(self.habitat_path, "habitat-lab/habitat-lab/habitat/config/benchmark/nav/", self.task, self.task + "_" + self.dataset + ".yaml")
        elif self.task == "vln_r2r":
            config_path = os.path.join(self.habitat_path, "habitat-lab/habitat-lab/habitat/config/benchmark/nav/", self.task + ".yaml")
        elif self.task == "eqa":
            config_path = os.path.join(self.habitat_path, "habitat-lab/habitat-lab/habitat/config/benchmark/nav/", self.task + "_" + self.dataset + ".yaml")
        elif self.task in self.rearranges:
            config_path = os.path.join(self.habitat_path, "habitat-lab/habitat-lab/habitat/config/benchmark/rearrange/", self.task + ".yaml")
        else:
            raise NotImplementedError("There is no task named " + task)
        self.device = device
        self.batch_size = batch_size
        self.config = habitat.get_config(config_path)

        if config is not None:
            with habitat.config.read_write(self.config):
                if "agent" in config.keys():
                    if "forward_step_size" in config["agent"].keys():
                        self.config.habitat.simulator.forward_step_size = config["agent"]["forward_step_size"]
                    if "turn_angle" in config["agent"].keys():
                        self.config.habitat.simulator.turn_angle = config["agent"]["turn_angle"]
            
        with habitat.config.read_write(self.config):
            self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = self.device
            self.config.habitat.dataset.split = split

        current_dir = os.getcwd()
        os.chdir(os.path.join(self.habitat_path, "habitat-lab"))
        if batch_size == 1:
            self.env = habitat.Env(config=self.config)
        else:
            #TODO Support multi batch process
            raise NotImplementedError("This library has not support batch_size > 1 yet.")
        os.chdir(current_dir)
        

    def get_config(self) -> dict:

        dict_conf = {
            "agent": {
                "forward_step_size": self.config.habitat.simulator.forward_step_size,
                "turn_angle": self.config.habitat.simulator.turn_angle,
                "radius": self.config.habitat.simulator.agents.main_agent.radius,
                "height": self.config.habitat.simulator.agents.main_agent.height,
            },
        }
        if "rgb_sensor" in self.config.habitat.simulator.agents.main_agent.sim_sensors.keys():
            dict_conf["rgb"] = {
                "width": self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width,
                "height": self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height,
                "hfov": self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov,
                "position": self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position,
            }
        if "depth_sensor" in self.config.habitat.simulator.agents.main_agent.sim_sensors.keys():
            dict_conf["depth"] = {
                "width": self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width,
                "height": self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height,
                "hfov": self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov,
                "min_depth": self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth,
                "max_depth": self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth,
                "position": self.config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position,
            }
        elif "head_depth_sensor" in self.config.habitat.simulator.agents.main_agent.sim_sensors.keys():
            dict_conf["depth"] = {
                "width": self.config.habitat.simulator.agents.main_agent.sim_sensors.head_depth_sensor.width,
                "height": self.config.habitat.simulator.agents.main_agent.sim_sensors.head_depth_sensor.height,
                "hfov": self.config.habitat.simulator.agents.main_agent.sim_sensors.head_depth_sensor.hfov,
                "min_depth": self.config.habitat.simulator.agents.main_agent.sim_sensors.head_depth_sensor.min_depth,
                "max_depth": self.config.habitat.simulator.agents.main_agent.sim_sensors.head_depth_sensor.max_depth,
                "position": self.config.habitat.simulator.agents.main_agent.sim_sensors.head_depth_sensor.position,
            }
        if self.task in self.rearranges:
            dict_conf["sampling_interval"] = 0.1
            dict_conf["agent"]["action_type"] = "continuous"
            dict_conf["agent"]["max_velocity"] = dict_conf["agent"]["forward_step_size"] / dict_conf["sampling_interval"]
            dict_conf["agent"]["max_angular_velocity"] = dict_conf["agent"]["turn_angle"] / dict_conf["sampling_interval"]
        else:
            dict_conf["agent"]["action_type"] = "discrete"
        
        return dict_conf

    def num_episodes(self) -> int:
        return len(self.env.episodes)

    def step(self, inputs) -> (dict, dict):
        current_dir = os.getcwd()
        os.chdir(os.path.join(self.habitat_path, "habitat-lab"))
        if self.batch_size == 1:
            if not self.task in self.rearranges:
                action = inputs["action"][0, 0].item()
                if action == -1:
                    action = 0
                if self.task == "eqa" and action > 0:
                    action -= 1
                obs = self.env.step(action=int(action))
                info = self.env.get_metrics()
                obs, info = self.refine_format(obs, info)
                obs["prev_action"] = inputs["action"].clone()
            else:
                cmd_vel = inputs["cmd_vel"][0].cpu().numpy()
                command = {
                    "action": ("arm_action", "base_velocity"),
                    "action_args": {
                        "arm_action": np.random.rand(7),
                        "grip_action": np.random.rand(1),
                        "base_vel": cmd_vel,
                        "REARRANGE_STOP": 0,
                    }
                }
                obs = self.env.step(action=command)
                info = self.env.get_metrics()
                obs, info = self.refine_format(obs, info)
                obs["prev_action"] = inputs["action"].clone()
        else:
            #TODO Support multi batch process
            raise NotImplementedError("This library has not support batch_size > 1 yet.")
        os.chdir(current_dir)
        return obs, info

    def reset(self) -> (dict, dict):
        current_dir = os.getcwd()
        os.chdir(os.path.join(self.habitat_path, "habitat-lab"))
        obs = self.env.reset()
        info = self.env.get_metrics()
        os.chdir(current_dir)
        return self.refine_format(obs, info)

    def is_episode_over(self) -> bool:
        return self.env.episode_over

    def refine_format(self, obs, info) -> (dict, dict):

        #print(obs)
        #print(info)
        refine_obs = {}
        if "rgb" in obs.keys():
            refine_obs["rgb"] = torch.Tensor(obs["rgb"]).permute(2, 0, 1).to(self.device).unsqueeze(0)
        if "depth" in obs.keys():
            refine_obs["depth"] = torch.Tensor(obs["depth"]).permute(2, 0, 1).to(self.device).unsqueeze(0)
        if "semantic" in obs.keys():
            refine_obs["semantic"] = torch.Tensor(obs["semantic"]).permute(2, 0, 1).to(self.device).unsqueeze(0)
        if "gps" in obs.keys():
            refine_obs["position2d"] = torch.Tensor(obs["gps"]).to(self.device).unsqueeze(0)
        if "compass" in obs.keys():
            refine_obs["rotation2d"] = torch.Tensor(obs["compass"]).to(self.device).unsqueeze(0)
        if "pointgoal_with_gps_compass" in obs.keys():
            refine_obs["goal2d"] = torch.Tensor(obs["pointgoal_with_gps_compass"]).to(self.device).unsqueeze(0)
        if "instruction" in obs.keys():
            refine_obs["instruction"] = [obs["instruction"]["text"]]
            refine_obs["instruction_token"] = torch.Tensor(obs["instruction"]["tokens"]).to(self.device).unsqueeze(0)
        if "imagegoal" in obs.keys():
            refine_obs["imagegoal"] = torch.Tensor(obs["imagegoal"]).permute(2, 0, 1).to(self.device).unsqueeze(0)
        if self.task == "objectnav":
            refine_obs["objectgoal"] = [self.env.current_episode.object_category]
            text = "Find a " + refine_obs["objectgoal"][0] + "."
            refine_obs["instruction"] = [text]
        if self.task == "eqa":
            refine_obs["question"] = [info["episode_info"]["question"].question_text]

        # for rearrange
        if "head_depth" in obs.keys():
            refine_obs["head_depth"] = torch.Tensor(obs["head_depth"]).permute(2, 0, 1).to(self.device).unsqueeze(0)
        if "obj_start_sensor" in obs.keys():
            refine_obs["obj_start"] = torch.Tensor(obs["obj_start_sensor"]).to(self.device).unsqueeze(0)
        if "obj_goal_sensor" in obs.keys():
            refine_obs["obj_goal"] = torch.Tensor(obs["obj_goal_sensor"]).to(self.device).unsqueeze(0)
        if "ee_pos" in obs.keys():
            refine_obs["ee_pos"] = torch.Tensor(obs["ee_pos"]).to(self.device).unsqueeze(0)
        if "relative_resting_position" in obs.keys():
            refine_obs["relative_resting_position"] = torch.Tensor(obs["relative_resting_position"]).to(self.device).unsqueeze(0)
        if "obj_start_gps_compass" in obs.keys():
            refine_obs["obj_start_gps_compass"] = torch.Tensor(obs["obj_start_gps_compass"]).to(self.device).unsqueeze(0)
        if "obj_goal_gps_compass" in obs.keys():
            refine_obs["obj_goal_gps_compass"] = torch.Tensor(obs["obj_goal_gps_compass"]).to(self.device).unsqueeze(0)

        # get agent position
        agent_state = self.env.sim.get_agent_state()
        source_position = np.array(self.env.current_episode.start_position, dtype=np.float32)
        source_rotation = quaternion_from_coeff(self.env.current_episode.start_rotation)
        direction_vector = agent_state.position - source_position
        relative_position = quaternion_rotate_vector(source_rotation.inverse(), direction_vector)
        relative_rotation = source_rotation.inverse() * agent_state.rotation

        refine_info = {
            "position": torch.Tensor(relative_position).to(self.device).unsqueeze(0),
            "rotation": torch.Tensor(relative_rotation.components).to(self.device).unsqueeze(0),
            "source_position": torch.Tensor(source_position).to(self.device).unsqueeze(0),
            "source_rotation": torch.Tensor(source_rotation.components).to(self.device).unsqueeze(0),
        }

        # get info about goals
        if hasattr(self.env.current_episode, "goals"):
            num_goal = len(self.env.current_episode.goals)
            goal_positions = []
            for n in range(num_goal):
                goal_position = np.array(self.env.current_episode.goals[n].position, dtype=np.float32)
                goal_direction = goal_position - source_position
                relative_goal_position = quaternion_rotate_vector(source_rotation.inverse(), goal_direction)    
                goal_positions.append(torch.Tensor(relative_goal_position).to(self.device).unsqueeze(0))
            goal_position = torch.cat(goal_positions, dim=0).unsqueeze(0)
            refine_info["goal_position"] = goal_position

        if "distance_to_goal" in info.keys():
            refine_info["distance_to_goal"] = torch.Tensor([info["distance_to_goal"]]).to(self.device).unsqueeze(0)
        if "success" in info.keys():
            refine_info["success"] = torch.Tensor([info["success"]]).to(self.device).unsqueeze(0)
        if "spl" in info.keys():
            refine_info["spl"] = torch.Tensor([info["spl"]]).to(self.device).unsqueeze(0)
        if "softspl" in info.keys():
            refine_info["softspl"] = torch.Tensor([info["softspl"]]).to(self.device).unsqueeze(0)
        if self.task == "eqa":
            refine_info["answer_accuracy"] = torch.Tensor([info["answer_accuracy"]]).to(self.device).unsqueeze(0)
            refine_info["answer"] = info["episode_info"]["question"].answer_text

        # for rearrange
        if "articulated_agent_force" in info.keys():
            refine_info["articulated_agent_force"] = info["articulated_agent_force"]
        if "force_terminate" in info.keys():
            refine_info["force_terminate"] = info["force_terminate"]
        if "robot_collisions" in info.keys():
            refine_info["robot_collisions"] = info["robot_collisions"]
        if "object_to_goal_distance" in info.keys():
            refine_info["object_to_goal_distance"] = info["object_to_goal_distance"]
        if "num_steps" in info.keys():
            refine_info["num_steps"] = info["num_steps"]
        if "ee_to_object_distance" in info.keys():
            refine_info["ee_to_object_distance"] = info["ee_to_object_distance"]
        if "does_want_terminate" in info.keys():
            refine_info["does_want_terminate"] = info["does_want_terminate"]
        if "composite_success" in info.keys():
            refine_info["composite_success"] = info["composite_success"]
        if "bad_called_terminate" in info.keys():
            refine_info["bad_called_terminate"] = info["bad_called_terminate"]
        if "did_violate_hold_constraint" in info.keys():
            refine_info["did_violate_hold_constraint"] = info["did_violate_hold_constraint"]
        if "move_obj_reward" in info.keys():
            refine_info["move_obj_reward"] = info["move_obj_reward"]

        return refine_obs, refine_info
