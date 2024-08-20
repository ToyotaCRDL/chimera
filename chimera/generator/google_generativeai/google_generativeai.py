#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import os
import types
import google.generativeai as genai
from google.generativeai import protos
from google.generativeai.types import content_types
from chimera. generator import Generator
import torch
from PIL import Image

class GoogleGenerativeAI(Generator):
    def __init__(self, config=None, model="gemini-1.5-flash", device=0, batch_size=1, 
        verbose=False,
        max_messages=1000,
        automatic_function_calling=True,
        **kwargs):
        
        if batch_size != 1:
            raise Exception("batch_size != 1 is not supported.")

        self.batch_size = batch_size
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model)
        if automatic_function_calling:
            self.chat_session = self.model.start_chat(enable_automatic_function_calling=True)
            self.keep_parts = []
        else:
            self.chat_session = None
        self.histories = []
        for b in range(self.batch_size):
            self.histories.append([])
        self.functions = []
        self.images = []

        self.verbose = verbose
        self.max_messages = max_messages

    def reset(self, batch=None):
        if batch is None:
            self.histories = []
            for b in range(self.batch_size):
                self.histories.append([])
            self.functions = []
        else:
            self.histories[batch] = []

    def add_function(self, function):
        if isinstance(function, types.FunctionType):
            if self.model._tools is None:
                self.model._tools = content_types.to_function_library(function)
            else:
                tool = content_types._make_tool(function)
                self.model._tools._tools.append(tool)
        elif isinstance(function, list):
            for func in function:
                if isinstance(func, types.FunctionType):
                    if self.model._tools is None:
                        self.model._tools = content_types.to_function_library(function)
                    else:
                        tool = content_types._make_tool(function)
                        self.model._tools._tools.append(tool)
                elif isinstance(func, dict):
                    raise Exception("Not implemented yet")
        elif isinstance(function, dict):
            raise Exception("Not implemented yet")

    def add_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.to(torch.float) / 255.0
            img = ToPILImage()(image)
        elif isinstance(image, str):
            img = Image.open(image)
        else:
            raise NotImplementedError("Input image format is not supported.")    

        self.keep_parts.append(img)
        
    def add_message(self, message):

        # Add message to histories
        if type(message) == str:
            message = {
                "role":"user",
                "parts": [message],
            }
            self.histories[0].append(message)
        elif type(message) == dict:
            self.histories[0].append(message)
        elif type(message) == list:
            for b in range(self.batch_size):
                if type(message[b]) == str:
                    msg = {
                        "role":"user",
                        "parts":[message[b]],
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

    def __call__(self, prompt=None, **kwargs):
        return self.chat(message=prompt, **kwargs)

    def chat(self,
            message=None,
            function=None,
            image=None,
            tool_choice="auto",
            **kwargs):

        if "function_call" in kwargs.keys():
            tool_choice = kwargs["function_call"]

        if self.chat_session is not None:
            if isinstance(message, str):
                message = {
                    "role":"user",
                    "parts":[message],
                }
            for part in self.keep_parts:
                message["parts"].append(part)
            response = self.chat_session.send_message(message)
        else:
            if message is not None:
                self.add_message(message)

            if function is not None:
                self.add_function(function)

            if image is not None:
                self.add_image(image)
      
            response = self.model.generate_content(self.histories[0])

        if self.verbose:
            print(response)

        # add interface corresponding to OpenAI API
        response.role = response.candidates[0].content.role
        response.content = response.candidates[0].content.parts
        self.add_message({
            "role": response.role,
            "parts": response.content,
            })

        return response

