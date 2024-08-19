#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import chimera
import torch
import pytorch3d.transforms as p3dt
import argparse
from datetime import datetime

class ObjectNavAgent:
    def __init__(self, config,
        device=0,
        detector="YOLOv8",
        confthresh=0.7,
        ):
        
        self.config = config
        self.device = device
        self.det = chimera.create_detector(self.config, name=detector)
        self.config.update(self.det.get_config())
        self.loc = chimera.create_localizer(self.config)
        self.mapper = chimera.create_mapper(self.config)
        self.sem_mapper = chimera.create_mapper(self.config, name="SemanticMapper", conf_thresh=confthresh)
        self.nav = chimera.create_navigator(self.config, goal_radius=1.0)
        
    def get_config(self):
        return self.config
    
    def reset(self):
        self.loc.reset()
        self.det.reset()
        self.mapper.reset()
        self.sem_mapper.reset()
        self.nav.reset()
        self.prev_position2d = None
        
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
            obs.update(self.det.detect(obs))
            obs.update(self.sem_mapper.add(obs))
            
            if not "objectgoal" in obs.keys():
                obs["objectgoal"] = [args.objectgoal]
            goal2d_xy = self.sem_mapper.GetNearestObjectGoal(obs["objectgoal"], obs["position2d"])
            
            if goal2d_xy[0] is not None:
                obs["goal2d_xy"] = goal2d_xy[0].unsqueeze(0)
                inputs = self.nav.act(obs)
            else:
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
        config = chimera.create_config()
        sim = chimera.create_simulator(name=args.simulator, config=config, task="objectnav", dataset=args.dataset, split=args.split)
        config.update(sim.get_config())
        agent = ObjectNavAgent(config, detector=args.detector, confthresh=args.confthresh)
        config.update(agent.get_config())
        vis = chimera.create_visualizer(config, show=True, height=240, ego_size=240, show_rgbmap=True)
        successes = []
        spls = []
        
        for episode in range(sim.num_episodes()):
            
            agent.reset()

            goal2d_xy = None
            timestamp = 0.0
            
            obs, info = sim.reset()
            obs["timestamp"] = [timestamp]
            
            inputs, obs = agent.act(obs)
            obs.update(inputs)
            if "objects" in obs.keys():
                if "names" in obs["objects"].keys():
                    config["objects"]["names"] = obs["objects"]["names"]            

            vis.visualize(obs)
            
            while not sim.is_episode_over():
            
                obs, info = sim.step(inputs)
                timestamp += 1.0
                obs["timestamp"] = [timestamp]
                
                inputs, obs = agent.act(obs)
                obs.update(inputs)
                if "objects" in obs.keys():
                    if "names" in obs["objects"].keys():
                        config["objects"]["names"] = obs["objects"]["names"]
                    
                vis.visualize(obs)
                
            if args.save:
                strtime = datetime.now().strftime("%Y%m%d_%H%M%S")
                vis.save_video("results", "demo_objnav_" + str(episode) + "_" + strtime)
                
            # keep result
            if "success" in info.keys():
                successes.append(info["success"])
                if info["success"][0, 0].item() == 1:
                    print(str(episode) + " SUCCESS")
                else:
                    print(str(episode) + " FAIL")
            if "spl" in info.keys():
                spls.append(info["spl"])
    
    #print results
    if len(successes) > 0:
        success = torch.cat(successes, dim=0).mean()
        print("Success = " + str(success.tolist()))
    if len(spls) > 0:
        spl = torch.cat(spls, dim=0).mean()
        print("SPL = " + str(spl.tolist()))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="demo_objectnav",
        usage="Demonstration of Object-goal Navigation",
        add_help=True,
        )
    parser.add_argument("-sim", "--simulator", help="name of the simulator", default="Habitat")
    parser.add_argument("-data", "--dataset", help="name of the dataset", default="hm3d")
    parser.add_argument("-split", "--split", help="split of the dataset", default="val_mini")
    parser.add_argument("-det", "--detector", help="detection method", default="YOLOv8")
    parser.add_argument("-th", "--confthresh", help="threshold of confidence", default=0.7)
    parser.add_argument("-objgoal", "--objectgoal", help="default object-goal", default="chair")
    parser.add_argument("-s", "--save", help="save the mp4 videos", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
            
        
        
