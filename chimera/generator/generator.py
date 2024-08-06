#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import abc

class Generator(metaclass=abc.ABCMeta):
    """
    This class is an abstract class which provides interface of generators.
    """

    @abc.abstractmethod
    def __call__(self, prompt, **kwargs):
        """
        Generate something.

        Args:
            prompt: the prompt to generate something.

        Returns:
            outputs: generated objects.
        """
        raise NotImplementedError()

# For spcific generator
def create_llm(config=None, name="gpt-4o", device=0, **kwargs):
    return create_generator(config, name, device, **kwargs)

# For spcific generator
def create_image_generator(config=None, name="StableDiffusion", device=0, **kwargs):
    return create_generator(config, name, device, **kwargs)

def create_generator(config=None, name="gpt-4o", device=0, **kwargs):
    """
    Create an Generator.

    Args:
        config (dict) The config parameters for the environment.: 
        name (str): The name of object detection method (e.g., `gpt-4o`)
        device (int): The number of cuda device.

    Returns:
        Generator: The instance of the defined Generator.
    """
    print("Generator = " + name)
    if name == "gpt" or name == "GPT" or name == "openai" or name == "OpenAI":
        model = kwargs["model"] if "model" in kwargs.keys() else "gpt-4o"
        from chimera.generator.openai import OpenAI
        return OpenAI(config, model, device, **kwargs)
    elif name.startswith("gpt-"):
        from chimera.generator.openai import OpenAI
        return OpenAI(config, name, device, **kwargs)
    elif name == "gemini" or name == "Gemini":
        model = kwargs["model"] if "model" in kwargs.keys() else "gemini-1.5-pro"
        from chimera.generator.google_generativeai import GoogleGenerativeAI
        return GoogleGenerativeAI(config, model, device **kwargs)
    elif name.startswith("gemini-"):
        from chimera.generator.google_generativeai import GoogleGenerativeAI
        return GoogleGenerativeAI(config, name, device, **kwargs)
    elif name == "stablediffusion" or name == "StableDiffusion":
        from chimera.generator.stablediffusion import StableDiffusion
        return StableDiffusion(config, device, **kwargs)
    else:
        raise NotImplementedError("There is no Generator named:" + name)

