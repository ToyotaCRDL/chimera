#!/usr/bin/env python3
# Copyright (c) Toyota Central R&D Labs., Inc.
# All rights reserved.

import chimera
import torch

def main():

    with torch.no_grad():
        device = 0
        sim = chimera.create_simulator(name="MetaWorld", dataset="MT1", task="drawer-close-v2", split="train", num_episodes_per_task=10, goal_visible=True, camera="corner2", dt=0.05)
        config = chimera.create_config()
        config.update(sim.get_config())
        vis = chimera.create_visualizer(config)
        successes = []

        print("# of episodes = " + str(sim.num_episodes()))
        for episode in range(sim.num_episodes()):
            obs, info = sim.reset()
            vis.reset()
            vis.visualize(obs)

            while not sim.is_episode_over():
                inputs = {
                    "cmd_ee_pos": 2 * torch.rand(1, 3).to(device) - 1,
                    "cmd_gripper": 2 * torch.rand(1, 1).to(device) - 1,
                }
                obs, info = sim.step(inputs)
                vis.visualize(obs)
                
            successes.append(info["success"])
            if info["success"][0, 0].item() == 1:
                print(str(episode) + " SUCCESS")
            else:
                print(str(episode) + " FAIL")

        success = torch.cat(successes, dim=0).mean()
        print("Success = " + str(success.tolist()))
    
if __name__ == "__main__":
    main()    
