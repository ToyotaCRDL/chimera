#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import os
import openai
from chimera.generator import Generator

import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import io
import base64
import inspect
import typing
import json

def format_type(annotation) -> str:
    if annotation is inspect._empty:
        return 'unknown'
    elif isinstance(annotation, type):
        return annotation.__name__
    else:
        return str(annotation)

class OpenAI(Generator):
    def __init__(self, config=None, model="gpt-4o", device=0, batch_size=1, 
        api_key=None,
        verbose=False,
        max_messages=1000,
        automatic_function_calling=True,
        **kwargs):

        if batch_size != 1:
            raise Exception("batch_size != 1 is not supported.")
        self.batch_size = batch_size
        self.model = model
        openai.api_key = os.environ["OPENAI_API_KEY"] if api_key is None else api_key
        self.histories = []
        for b in range(self.batch_size):
            self.histories.append([])
        self.functions = []
        self.callables = {}
        self.images = []

        self.verbose = verbose
        self.max_messages = max_messages
        self.automatic_function_calling = automatic_function_calling

    def reset(self, batch=None):
        if batch is None:
            self.histories = []
            for b in range(self.batch_size):
                self.histories.append([])
            self.functions = []
            self.callables = {}
        else:
            self.histories[batch] = []

    
    def add_callable(self, function):
        func_dict = {}
        func_dict["name"] = function.__name__
        func_dict["description"] = inspect.getdoc(function)
        signature = inspect.signature(function)
        parameters = signature.parameters
        func_dict["parameters"] = {
            "type": "object",
            "properties": {},
        }
    
        required_params = []
        for param_name, param in parameters.items():
            func_dict["parameters"]["properties"][param_name] = {
                "type": "string", #format_type(param.annotation),
                "description": param_name,
            }
            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)
        func_dict["parameters"]["required"] = required_params
        self.functions.append(func_dict)
        self.callables[function.__name__] = function

    def add_function(self, function):
        if isinstance(function, typing.Callable):
            self.add_callable(function)
        elif isinstance(function, dict):
            self.functions.append(function)
            self.automatic_function_calling = False
        elif isinstance(function, list):
            for func in function:
                if isinstance(func, typing.Callable):
                    self.add_callable(func)
                elif isinstance(func, dict):
                    self.functions.append(func)
                    self.automatic_function_calling = False
        else:
            raise Exception("Unexpected type of function is input.")

    def add_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.to(torch.float) / 255.0
            image = ToPILImage()(image)

            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG")
                image_data = buffer.getvalue()
            base64_image = base64.b64encode(image_data).decode("utf-8")

        elif isinstance(image, str):
            with open(image, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            raise NotImplementedError("Input image format is not supported.")    

        message = {
            "role":"user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }

        self.histories[0].append(message)
        

    def add_message(self, message):

        # Add message to histories
        if type(message) == str:
            message = {
                "role":"user",
                "content": message,
            }
            self.histories[0].append(message)
        elif type(message) == dict:
            self.histories[0].append(message)
        elif type(message) == list:
            for b in range(self.batch_size):
                if type(message[b]) == str:
                    msg = {
                        "role":"user",
                        "content":message[b],
                    }
                    self.histories[b].append(msg)
                elif type(message[b]) == dict:
                    self.histories[b].append(message[b])
        
        for b in range(self.batch_size):
            while(len(self.histories[b]) > self.max_messages):
                found = False
                for i, hist in enumerate(self.histories[b]):
                    if hist["role"] != "system":
                        self.histories[b].pop(i)
                        found = True
                        break
                if not found:
                    raise Exception("Too many system messages more than max_messages.")

    def __call__(self, prompt, **kwargs):
        return self.chat(message=prompt, **kwargs)

    def chat(self, 
            message=None, 
            function=None, 
            image=None, 
            tool_choice="auto", 
            **kwargs):

        if "function_call" in kwargs.keys():
            tool_choice = kwargs["function_call"]

        if message is not None:
            self.add_message(message)

        if function is not None:
            self.add_function(function)

        if image is not None:
            self.add_image(image)

        if self.batch_size == 1:
            response = self.chat_completion(self.histories[0], tool_choice, **kwargs)
            msg = response.choices[0].message
            self.add_message(msg.to_dict())

            if self.automatic_function_calling:
                while msg.tool_calls is not None:
                    for tool_call in msg.tool_calls:
                        func_call = tool_call.function

                        func_name = func_call.name
                        func_args = json.loads(func_call.arguments)
                        func_res = self.callables[func_name](**func_args)

                        if isinstance(func_res, torch.Tensor):
                            tool_res = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": func_call.name,
                                "content": "The response image is:",
                            }
                            for b in range(func_res.shape[0]):
                                llm.add_image(func_res[b])
                        else:
                            tool_res = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": func_call.name,
                                "content": str(func_res),
                            }
                            self.add_message(tool_res)

                    response = self.chat_completion(self.histories[0], tool_choice, **kwargs)
                    msg = response.choices[0].message
                    self.add_message(msg.to_dict())

            return msg
            
    def chat_completion(self, messages, tool_choice="auto", **kwargs):
        #
        # memo: message format
        # messages = [
        #   {
        #       "role":"user",
        #       "content":message,
        #   }]
        #
        if "function_call" in kwargs.keys():
            tool_choice = kwargs["function_call"]

        if self.verbose:
            print("Functions:" + str(self.functions))
            print("Message:" + str(messages))

        if len(self.functions) > 0:
            tools = []
            for func in self.functions:
                tool = {
                    "type": "function",
                    "function": func,
                }
                tools.append(tool)

            response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
        else:
            response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )

        if self.verbose:
            print(response)

        return response
