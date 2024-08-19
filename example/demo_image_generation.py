#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import os
import argparse
import chimera

def main(args):
    sd = chimera.create_image_generator()
    print("Type a prompt to generate image: (Enter to exit)")
    prompt = input()
    while len(prompt) > 0:
        print("prompt = `" + prompt + "'")
        images = sd(prompt)
        images[0].show()
        if args.save:
            output_dir = "results"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_file = os.path.join(output_dir, prompt + ".png")
            images[0].save(save_file)
            print("Save `" + save_file + "'")
        print("Enter a new prompt to continue: (Enter to exit)")
        prompt = input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="demo_image_generation",
        usage="Demonstration of LLM on chimera",
        add_help=True,
        )
    parser.add_argument("-n", "--name", help="name of the image generation", default="stable_diffusion")
    parser.add_argument("-s", "--save", help="save image", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
