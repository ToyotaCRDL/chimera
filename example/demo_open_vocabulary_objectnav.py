#!/usr/bin/env python3
# Copyright (c) Toyota Central R&D Labs., Inc.
# All rights reserved.

import chimera
import torch
import argparse
from datetime import datetime
import os
import csv

class ObjectNavAgent:
    def __init__(self, config, device=0):
        
        self.config = config
        self.device = device
        self.loc = chimera.create_localization(self.config)
        self.mapper = chimera.create_mapper(self.config)
        self.clip_mapper = chimera.create_mapper(self.config, name="CLIPMapper")
        self.nav = chimera.create_navigation(self.config, goal_radius=1.0)
        self.initial_thresh = 0.27
        
    def get_config(self):
        return self.config
    
    def reset(self):
        self.loc.reset()
        self.mapper.reset()
        self.clip_mapper.reset()
        self.nav.reset()
        self.prev_position2d = None
        self.sim_thresh = self.initial_thresh
        
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
            obs.update(self.clip_mapper.add(obs))
            
            if not "objectgoal" in obs.keys():
                obs["objectgoal"] = [args.objectgoal]
            outputs = self.clip_mapper.find(obs, similarity_thresh=self.sim_thresh)
            goal2d_xy = outputs["goal2d_xy"]
            obs = chimera.expand_inputs(obs)
            
            if goal2d_xy[0] is not None:
                obs["goal2d_xy"] = goal2d_xy
                inputs = self.nav.act(obs)
                while inputs["action"][0, 0] == -1 and self.sim_thresh > 0: # Fail
                    obs["goal2d_xy"] = -50 * torch.ones(1, 2).to(self.device)
                    inputs = self.nav.act(obs)
                    if inputs["action"][0, 0] == -1:
                        self.sim_thresh -= 0.001
                        outputs = self.clip_mapper.find(obs, similarity_thresh=self.sim_thresh)
                        goal2d_xy = outputs["goal2d_xy"]
                        if goal2d_xy[0] is not None:
                            obs["goal2d_xy"] = goal2d_xy
                            inputs = self.nav.act(obs)
            else:
                obs["goal2d_xy"] = -50 * torch.ones(1, 2).to(self.device)
                inputs = self.nav.act(obs)
                while inputs["action"][0, 0] == -1 and self.sim_thresh > 0: # Fail
                    self.sim_thresh -= 0.001
                    outputs = self.clip_mapper.find(obs, similarity_thresh=self.sim_thresh)
                    goal2d_xy = outputs["goal2d_xy"]
                    if goal2d_xy[0] is not None:
                        obs["goal2d_xy"] = goal2d_xy
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
        config = chimera.create_config()
        sim = chimera.create_simulator(name=args.simulator, config=config, task=args.task, dataset=args.dataset, split=args.split)
        config.update(sim.get_config())
        agent = ObjectNavAgent(config)
        config.update(agent.get_config())
        vis = chimera.create_visualizer(config)
        
        for episode in range(1):#range(sim.num_episodes()):

            images = []
            agent.reset()
            vis.reset()

            goal2d_xy = None
            timestamp = 0.0
            
            obs, info = sim.reset()
            obs["timestamp"] = [timestamp]
            obs["instruction"] = [""]
            
            inputs, obs = agent.act(obs)
            obs.update(inputs)
            
            vis.visualize(obs)
            
            while not sim.is_episode_over():

                print("Input first object-goal:")
                objectgoal = input()
  
                while objectgoal:            

                    agent.sim_thresh = agent.initial_thresh
                    inputs["action"][0, 0] = 1; # Forward

                    while not sim.is_episode_over() and inputs["action"][0, 0] > 0:
                        obs, info = sim.step(inputs)
                        timestamp += 1.0
                        obs["timestamp"] = [timestamp]
                        obs["objectgoal"] = [objectgoal]
                        obs["instruction"] = ["Find " + objectgoal]
                        
                        inputs, obs = agent.act(obs)
                        obs.update(inputs)
                            
                        vis.visualize(obs)

                    if inputs["action"][0, 0] == 0:
                        print("DONE!!")
                    else:
                        print("Episode is over...")
                        break

                    print("Input next object-goal:")
                    objectgoal = input()                

                if not sim.is_episode_over():
                    inputs["action"][0, 0] = 0 # Stop
                    obs, info = sim.step(inputs)
                    timestamp += 1.0
                    obs["timestamp"] = [timestamp]

            if args.save:
                strtime = datetime.now().strftime("%Y%m%d_%H%M%S")
                vis.save_video("results", "demo_multion_" + str(episode) + "_" + strtime)
                                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="demo_open_vocabulary_objectnav",
        usage="Demonstration of Open Vocabulary Object-goal Navigation",
        add_help=True,
        )
    parser.add_argument("-sim", "--simulator", help="name of the simulator", default="Habitat")
    parser.add_argument("-task", "--task", help="task of the dataset", default="objectnav")
    parser.add_argument("-data", "--dataset", help="name of the dataset", default="hm3d")
    parser.add_argument("-split", "--split", help="split of the dataset", default="val_mini")
    parser.add_argument("-s", "--save", help="save the mp4 videos", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
            
        
        
