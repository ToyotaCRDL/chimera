#!/usr/bin/env python3
# Copyright (c) Toyota Central R&D Labs., Inc.
# All rights reserved.

from flask import Flask, render_template, jsonify, request, send_file
import io
import matplotlib.pyplot as plt
import chimera
import argparse
import torch
import numpy as np
from torchvision.transforms import ToPILImage
from PIL import Image
import json
import base64
from werkzeug.serving import WSGIRequestHandler

app = Flask(__name__)

# ArgParse
parser = argparse.ArgumentParser(
    prog="demo_chat_service",
    usage="Demonstration of Chat Service via Web",
    add_help=True,
    )
parser.add_argument("-n", "--name", help="name of the LLM", default="gpt-4o")
parser.add_argument("-port", "--port", help="port number", type=int, default=8888)
args = parser.parse_args()

# Initialize
with torch.no_grad():
    device = 0
    config = chimera.create_config()
    llm = chimera.create_llm(name=args.name, verbose="False")
    llm.reset()

@app.route('/')
def index():
    return render_template('chat_service.html', title="Demo Chat Service")

@app.route('/messages', methods=['POST'])
def messages():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400
    with torch.no_grad():
        llm.reset()
        chat_history = data
        for message in chat_history:
            chat_input = {"content": message["content"], "role": "user" if message["is_user"] else "assistant"}
            llm.add_message(chat_input)
        res = llm.chat()
    return jsonify({"content": res.content, "is_user": False}), 201

if __name__ == "__main__":
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host='0.0.0.0', debug=True, port=args.port, threaded=True)
