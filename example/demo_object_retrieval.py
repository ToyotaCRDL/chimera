#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import chimera
import os
import torch
import argparse

class ExplorationAgent:
    def __init__(self, config,
        device=0):
        
        self.config = config
        self.device = device
        self.loc = chimera.create_localization(self.config)
        self.mapper = chimera.create_mapper(self.config)
        self.rgb_mapper = chimera.create_mapper(self.config, name="RGBMapper")
        self.clip_mapper = chimera.create_mapper(self.config, name="CLIPMapper")
        self.nav = chimera.create_navigation(self.config, goal_radius=1.0)
        
    def get_config(self):
        return self.config
    
    def reset(self):
        self.loc.reset()
        self.mapper.reset()
        self.rgb_mapper.reset()
        self.clip_mapper.reset()
        self.nav.reset()
        self.prev_position2d = None
        self.sim_thresh = 0.30
        
    def act(self, obs):

        enable_pose = False
        if "position2d" in obs.keys():
            enable_pose = True
        else:
            obs.update(self.loc.track(obs))
            enable_pose = self.loc.is_warmuped()
        obs = chimera.expand_inputs(obs)
        
        if self.prev_position2d is not None and "prev_action" in obs.keys():
            obs["collision"] = chimera.detect_collision(self.config, obs["position2d"], self.prev_position2d, obs["prev_action"])

        if enable_pose:
            obs.update(self.mapper.add(obs))
            obs.update(self.rgb_mapper.add(obs))
            obs.update(self.clip_mapper.add(obs))
            
            obs["goal2d_xy"] = -50 * torch.ones(1, 2).to(self.device)
            inputs = self.nav.act(obs)
                
            if inputs["action"][0, 0] == 0: # done
                print("A " + obs["objectgoal"][0] + " is found!!")
            elif inputs["action"][0, 0] == -1: # Fail
                print("Failed!!")
                inputs["action"][0, 0] = 0 # Stop
                
        else:
            obs["map2d"] = 0.5 * torch.ones(1, 1, self.config["map2d"]["size"], self.config["map2d"]["size"]).to(self.device)
        
            inputs = {
                "action": torch.Tensor([[2]]).to(self.device), # Turn left
                "cmd_vel": torch.Tensor([[0.0, -0.1]]).to(self.device)
            }
        self.prev_position2d = obs["position2d"].clone()
            
        return inputs, obs
            
def main(args):

    with torch.no_grad():
        device = 0

        print("dataset: " + args.dataset)
        print("split: " + args.split)
        print("episode: " + str(args.episode))

        data_dir = args.data_dir + "/" + args.dataset + "_" + args.split + "_" + str(args.episode)
        config = chimera.create_config()

        # Do Exploration to Create Map
        if not os.path.exists(data_dir):
            print("Can not find map dir...")
            print("Start exploration to create map...")
            sim = chimera.create_simulator(name=args.simulator, config=config, task="objectnav", dataset=args.dataset, split=args.split)
            config.update(sim.get_config())
            agent = ExplorationAgent(config)
            config.update(agent.get_config())
            vis = chimera.create_visualizer(config, show=True, height=240, ego_size=240, show_rgbmap=True)

            for episode in range(sim.num_episodes()):

                obs, info = sim.reset()
                # Seek target episode
                if episode != args.episode:
                    continue

                agent.reset()
                timestamp = 0.0
                obs["timestamp"] = [timestamp]

                inputs, obs = agent.act(obs)
                obs.update(inputs)
                
                vis.visualize(obs)

                while not sim.is_episode_over():
                        
                    obs, info = sim.step(inputs)
                    timestamp += 1.0
                    obs["timestamp"] = [timestamp]
                    
                    inputs, obs = agent.act(obs)
                    obs.update(inputs)
                        
                    vis.visualize(obs)
                    
                os.makedirs(data_dir, exist_ok=True)
                agent.clip_mapper.save(data_dir + "/clip_map.pt")
                torch.save(obs["map2d"].cpu(), data_dir + "/map2d.pt")
                torch.save(obs["rgbmap2d"].cpu(), data_dir + "/rgbmap2d.pt")

        # Do Object Retrieval
        obs = {}
        clip_mapper = chimera.create_mapper(config, name="CLIPMapper")
        clip_mapper.load(data_dir + "/clip_map.pt")

        map_scale = config["map2d"]["scale"]
        max_x = torch.max(clip_mapper.points[0][:, 0])
        min_x = torch.min(clip_mapper.points[0][:, 0])
        max_z = torch.max(clip_mapper.points[0][:, 2])
        min_z = torch.min(clip_mapper.points[0][:, 2])
        mid_x = (max_x + min_x) / 2
        mid_z = (max_z + min_z) / 2
        range_x = (max_x - min_x) / map_scale
        range_z = (max_z - min_z) / map_scale
        max_range = int(torch.max(range_x, range_z).item()) // 2 * 2 + 50
        
        load_map2d = data_dir + "/map2d.pt"
        load_rgbmap2d = data_dir + "/rgbmap2d.pt"
        obs["map2d"] = torch.load(load_map2d).to(device)
        obs["rgbmap2d"] = torch.load(load_rgbmap2d).to(device)
        obs["position2d"] = torch.zeros(1, 2).to(device)
        obs["position2d"][0, 0] = - mid_z
        obs["position2d"][0, 1] = mid_x        

        vis = chimera.create_visualizer(config, show=True, height=960, ego_size=max_range, hide_ego=True, show_rgbmap=True)
        vis.visualize(obs)

        print("Input object name: (Enter to exit)")
        prompt = input()

        while len(prompt) > 0:

            obs["objectgoal"] = [prompt]
            sim_thresh = args.threshold
            outputs = clip_mapper.find(obs, similarity_thresh=sim_thresh)
            goal2d_xy = outputs["goal2d_xy"]
            while goal2d_xy[0] is None and sim_thresh > 0:
                sim_thresh -= 0.001
                outputs = clip_mapper.find(obs, similarity_thresh=sim_thresh)
                goal2d_xy = outputs["goal2d_xy"]
            obs["goal2d_xy"] = goal2d_xy
            print("similarity_thresh = " + f'{sim_thresh:.3g}')
            vis.visualize(obs)
        
            print("Input object name: (Enter to exit)")
            prompt = input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="demo_object_retrieval",
        usage="Demonstration of Object Retrieval using CLIPMapper",
        add_help=True,
        )
    parser.add_argument("-data_dir", "--data_dir", help="data dir", default="maps")
    parser.add_argument("-sim", "--simulator", help="name of the simulator", default="Habitat")
    parser.add_argument("-data", "--dataset", help="name of the dataset", default="hm3d")
    parser.add_argument("-split", "--split", help="split of the dataset", default="val_mini")
    parser.add_argument("-ep", "--episode", help="index of the episode", type=int, default=0)
    parser.add_argument("-thresh", "--threshold", type=float, help="max threshold of similarity", default=0.5)
    args = parser.parse_args()
    main(args)
