#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

import chimera
from chimera.simulator import Simulator
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
import math
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class Vizbot(Simulator):
    def __init__(self, config=None, device=0, batch_size=1, 
            node_name="Chimera",
            camera_info_topic="/camera/color/camera_info",
            rgb_topic="/camera/color/image_raw",
            depth_topic="/camera/aligned_depth_to_color/image_raw",
            out_topic="/cmd_vel",
            pose_topic=None,
            obj1_topic=None,
            goal_topic=None,
            **kwargs):
        
        if batch_size > 1:
            raise Exception("Cannot set batch_size > 1.")
        
        self.config = chimera.create_config()
        if config is not None:
            self.config.update(config)
        self.config["agent"]["radius"] = 2.0
        self.config["agent"]["action_type"] = "continuous"
        self.config["agent"]["max_velocity"] = 0.1
        self.config["agent"]["max_angular_velocity"] = 0.1
        self.config["sampling_interval"] = 1.0
        self.config["depth"]["min_depth"] = 0.1
        self.config["depth"]["max_depth"] = 5.0
        self.config["depth"]["position"] = [0.0, 0.255, -0.135]
        self.device = device
        
        # Flags
        self.obs_rgb = False
        self.obs_depth = False
        self.obs_pose = False
        self.obs_obj1 = False
        self.obs_goal = False
        
        # Initialize ROS
        rospy.init_node(node_name, anonymous=False)
        self.subscriber_camera_info = None
        if camera_info_topic is not None:
            self.subscriber_camera_info = rospy.Subscriber(camera_info_topic, CameraInfo, self.callback_camera_info)
        self.subscriber_rgb = None
        if rgb_topic is not None:
            self.subscriber_rgb = rospy.Subscriber(rgb_topic, Image, self.callback_rgb, queue_size=1)
        self.subscriber_depth = None
        if depth_topic is not None:
            self.subscriber_depth = rospy.Subscriber(depth_topic, Image, self.callback_depth, queue_size=1)
        self.subscriber_pose = None
        if pose_topic is not None:
            self.subscriber_pose = rospy.Subscriber(pose_topic, PoseStamped, self.callback_pose, queue_size=1)
        self.subscriber_obj1 = None
        if obj1_topic is not None:
            self.subscriber_obj1 = rospy.Subscriber(obj1_topic, PoseStamped, self.callback_obj1, queue_size=1)
        self.subscriber_goal = None
        if goal_topic is not None:
            self.subscriber_goal = rospy.Subscriber(goal_topic, PoseStamped, self.callback_goal, queue_size=1)
                
        self.publisher = rospy.Publisher(out_topic, Twist, queue_size=1)
        
        rospy.on_shutdown(self.shutdown_hook)
        
        self.wait_observation()
        
        self.done = False
        
    def shutdown_hook(self):
        if self.subscriber_camera_info is not None:
            self.subscriber_camera_info.unregister()
        if self.subscriber_rgb is not None:
            self.subscriber_rgb.unregister()
        if self.subscriber_depth is not None:
            self.subscriber_depth.unregister()
        if self.subscriber_pose is not None:
            self.subscriber_pose.unregister()
        if self.subscriber_obj1 is not None:
            self.subscriber_obj1.unregister()
        if self.subscriber_goal is not None:
            self.subscriber_goal.unregister()
        self.publisher.unregister()

    def callback_camera_info(self, msg):
        self.config["rgb"]["width"] = msg.width
        self.config["rgb"]["height"] = msg.height
        self.config["rgb"]["intrinsics"] = msg.K
        fov_x = 2 * math.atan2(msg.K[2], msg.K[0])
        fov_x_degrees = fov_x * (180.0 / math.pi)
        self.config["rgb"]["hfov"] = fov_x_degrees
        self.config["depth"]["width"] = msg.width
        self.config["depth"]["height"] = msg.height
        self.config["depth"]["intrinsics"] = msg.K
        self.config["depth"]["hfov"] = fov_x_degrees
        
    def callback_rgb(self, msg):
        image_data = msg.data
        np_arr = np.frombuffer(image_data, np.uint8)
        image = np_arr.reshape(msg.height, msg.width, -1)
        image_tensor = transforms.ToTensor()(image)
        self.rgb = image_tensor.to(self.device)
        self.obs_rgb = True
    
    def callback_depth(self, msg):
        image_data = msg.data
        np_arr = np.frombuffer(image_data, np.uint16)
        depth_image = np_arr.reshape(msg.height, msg.width, -1)
        depth_image = depth_image.astype(np.float32)
        depth_image_tensor = torch.from_numpy(depth_image)
        self.depth = depth_image_tensor.permute(2, 0, 1).to(self.device)
        self.obs_depth = True

    def callback_pose(self, msg):
        self.position = torch.Tensor([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]).to(self.device)
        self.rotation = torch.Tensor([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]) .to(self.device)
        self.obs_pose = True
    
    def callback_obj1(self, msg):
        self.obj1_pos = torch.Tensor([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]).to(self.device)
        self.obj1_rot = torch.Tensor([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]) .to(self.device)
        self.obs_obj1 = True

    def callback_goal(self, msg):
        self.goal_pos = torch.Tensor([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]).to(self.device)
        self.goal_rot = torch.Tensor([msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]) .to(self.device)
        self.obs_goal = True

    def get_config(self) -> dict:
        return self.config
        
    def num_episodes(self) -> int:
        return 1

    def wait_observation(self):
    
        while not rospy.is_shutdown() and ((self.subscriber_rgb is not None and not self.obs_rgb) or (self.subscriber_depth is not None and not self.obs_depth)):
            rospy.sleep(0.01)     

        obs = {}
        
        if self.obs_rgb:      
            self.obs_rgb = False
            obs["rgb"] = self.rgb.unsqueeze(0) * 255.0
        
        if self.obs_depth:
            self.obs_depth = False
            obs["depth"] = self.depth.unsqueeze(0)
            min_depth = self.config["depth"]["min_depth"] * 1000.0
            max_depth = self.config["depth"]["max_depth"] * 1000.0
            obs["depth"][obs["depth"] < min_depth] = min_depth
            obs["depth"][obs["depth"] > max_depth] = max_depth
            obs["depth"] = (obs["depth"] - min_depth) / (max_depth - min_depth)
        
        if self.obs_pose:
            obs["position"] = self.position.unsqueeze(0)
            obs["rotation"] = self.rotation.unsqueeze(0)

        if self.obs_obj1:
            obs["obj1_pos"] = self.obj1_pos.unsqueeze(0)
            obs["obj1_rot"] = self.obj1_rot.unsqueeze(0)
        
        if self.obs_goal:
            obs["goal_pos"] = self.goal_pos.unsqueeze(0)
            obs["goal_rot"] = self.goal_rot.unsqueeze(0)

        return obs

    def step(self, inputs) -> (dict, dict):
        
        obs = {}
        info = {}
        
        # publish action
        if inputs["action"][0, 0] == 0:
            cmd_vel = [0.0, 0.0]
            self.done = True
        else:
            cmd_vel = inputs["cmd_vel"][0].cpu().numpy()
        msg = Twist()
        msg.linear.x = cmd_vel[0]
        msg.angular.z = -cmd_vel[1]
        self.publisher.publish(msg)
        
        obs = self.wait_observation()
        
        return obs, info
        
        
    def reset(self) -> (dict, dict):
        
        obs = {}
        info = {}
        self.done = False
    
        obs = self.wait_observation()
    
        return obs, info
        
        
    def is_episode_over(self) -> bool:
        return self.done
        
    
