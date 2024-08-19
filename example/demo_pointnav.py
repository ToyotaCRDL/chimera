#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import argparse
import chimera
import torch
import random

def main(args):

    config = chimera.create_config()
    sim = chimera.create_simulator(dataset=args.dataset, task="pointnav", split=args.split)
    config.update(sim.get_config())
    if args.gtpose:
        print("localization = GT-pose")
    else:
        loc = chimera.create_localizer(config, name=args.localizer)
    mapper = chimera.create_mapper(name="DepthAndTrajectoryMapper", config=config, height_thresh_min=-0.3)
    nav = chimera.create_navigator(config)
    successes = []
    spls = []

    print("# of episodes = " + str(sim.num_episodes()))
    for episode in range(sim.num_episodes()):
        images = []
        if not args.gtpose:
            loc.reset()
        mapper.reset()
        nav.reset()
        obs, info = sim.reset()
        timestamp = 0.0
        obs["timestamp"] = [timestamp]

        if args.gtpose:
            obs["position"] = info["position"]
            obs["rotation"] = info["rotation"]
        else:
            res = loc.track(obs)
            obs.update(res)
        obs = chimera.expand_inputs(obs)
        if args.gtpose or loc.is_warmuped():
            obs.update(mapper.add(obs))
            inputs = nav.act(obs)
        else:
            obs["map2d"] = 0.5 * torch.ones(1, 1, config["map2d"]["size"], config["map2d"]["size"]).to("cuda:0")
            inputs = {}
            inputs["action"] = torch.Tensor([[random.choice(range(1, 3))]]).to(torch.int64)
        obs.update(inputs)
        im = chimera.visualize(config, obs, show=True)
        images.append(im)
        prev_position2d = obs["position2d"].clone()

        while not sim.is_episode_over():
            obs, info = sim.step(inputs)
            timestamp += 1.0
            obs["timestamp"] = [timestamp]

            if args.gtpose:
                obs["position"] = info["position"]
                obs["rotation"] = info["rotation"]
            else:
                res = loc.track(obs)
                obs.update(res)
            obs = chimera.expand_inputs(obs)
            obs["collision"] = chimera.detect_collision(config, obs["position2d"], prev_position2d, obs["prev_action"])
            if args.gtpose or loc.is_warmuped():
                obs.update(mapper.add(obs))
                inputs = nav.act(obs)
            else:
                inputs = {}
                inputs["action"] = torch.Tensor([[random.choice(range(1, 3))]]).to(torch.int64)
            obs.update(inputs)
            im = chimera.visualize(config, obs, show=True)
            if args.gtpose or loc.is_warmuped():
            	images.append(im)
            prev_position2d = obs["position2d"].clone()

        successes.append(info["success"])
        spls.append(info["spl"])
        if info["success"][0, 0].item() == 1:
            print(str(episode) + " SUCCESS")
        else:
            print(str(episode) + " FAIL")
        if args.save:
            chimera.images_to_video(images, "results", "demo_pointnav_" + args.dataset + "_" + args.split + "_" + str(episode))

    success = torch.cat(successes, dim=0).mean()
    spl = torch.cat(spls, dim=0).mean()
    print("Success = " + str(success.tolist()))
    print("SPL = " + str(spl.tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="demo_pointnav",
        usage="Demonstration of Point-goal Navigation",
        add_help=True,
        )
    parser.add_argument("-data", "--dataset", help="name of the dataset", default="habitat-test-scenes")
    parser.add_argument("-split", "--split", help="split of the dataset", default="val")
    parser.add_argument("-gtpose", "--gtpose", help="use gt pose", action="store_true", default=False)
    parser.add_argument("-loc", "--localizer", help="name of localizer", default="DroidSLAM")
    parser.add_argument("-s", "--save", help="save the mp4 videos", action="store_true", default=False)
    args = parser.parse_args()
    main(args)    
