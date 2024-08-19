#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import argparse
import chimera
from PIL import Image

def main(args):
    llm = chimera.create_llm(name=args.name, verbose=args.verbose)
    llm.add_image(args.image)
    image = Image.open(args.image)
    image.show()
    print("Please type a question about the image: (Enter to exit)")
    prompt = input()
    while len(prompt) > 0:
        res = llm.chat(prompt)
        print(res.content)
        prompt = input()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="demo_vqa",
        usage="Demonstration of VQA on chimera",
        add_help=True,
        )
    parser.add_argument("-n", "--name", help="name of the LLM", default="gpt-4o")
    parser.add_argument("-i", "--image", help="path of image file", default="example/example.jpg")
    parser.add_argument("-v", "--verbose", help="show the raw response from the LLM", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
