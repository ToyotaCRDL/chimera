#!/usr/bin/env python3
# Copyright (c) Toyota Central R&D Labs., Inc.
# All rights reserved.

import argparse
import chimera

def main(args):
    l2m = chimera.create_mapper(name="L2M")
    print("Please type a path message to store the L2M: (Enter to exit)")
    prompt = input()
    while len(prompt) > 0:
        res = l2m.add(prompt)
        print("Please type additional path messages to store the L2M: (Enter to finish)")
        prompt = input()

    print("Let's ask the path between two points to L2M!")
    while True:
        print("Please type name of start point: (Enter to exit)")
        start = input()
        if len(start) == 0:
            break

        print("Please type name of goal point: (Enter to exit)")
        goal = input()
        if len(goal) == 0:
            break

        res = l2m.get_path(start, goal)

        print(res["message"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="demo_llm",
        usage="Demonstration of LLM on chimera",
        add_help=True,
        )
    parser.add_argument("-n", "--name", help="name of the LLM", default="gpt-3.5-turbo-0613")
    parser.add_argument("-v", "--verbose", help="show the raw response from the LLM", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
