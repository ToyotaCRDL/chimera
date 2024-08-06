#!/usr/bin/env python3
# Copyright (C) 2024 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

from chimera.detector import Detector
import os
import sys

import cv2
import torch
import torchvision.transforms as transforms

detic_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Detic")
sys.path.insert(0, detic_dir)
centernet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Detic/third_party/CenterNet2")
sys.path.insert(0, centernet_dir)

from detectron2.config import get_cfg
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detectron2.engine.defaults import DefaultPredictor
from detic.predictor import get_clip_embeddings
from detic.modeling.utils import reset_cls_test

class Detic(Detector):
    def __init__(self, config=None, device=0, batch_size=1, **kwargs):
        self.classes = ["bed", "chair", "sofa", "toilet", "plant", "tv"]
        self.conf_thresh = 0.5
        if config is not None:
            if "objects" in config.keys():
                self.classes = list(config["objects"]["names"].values())
                self.conf_thresh = config["objects"]["conf_thresh"]
        if "conf_thresh" in kwargs.keys():
            self.conf_thresh = kwargs["conf_thresh"]
        self.classifier = get_clip_embeddings(self.classes)

        num_classes = len(self.classes)
        self.batch_size = batch_size
        self.device = device
        config_file = os.path.join(detic_dir, "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")

        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(config_file)
        
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.conf_thresh
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_thresh
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.conf_thresh
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
        cfg.freeze()

        current_dir = os.getcwd()
        os.chdir(detic_dir)

        self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, self.classifier, num_classes)

        os.chdir(current_dir)

    def get_config(self):
        config = {
            "objects": {
                "names": {},
                "conf_thresh": self.conf_thresh
            }
        }
        for i, c in enumerate(self.classes):
            config["objects"]["names"][i] = c
        return config

    def change_category(self, category):
        self.classes = category
        self.classifier = get_clip_embeddings(self.classes)
        num_classes = len(self.classes)
        reset_cls_test(self.predictor.model, self.classifier, num_classes)

    def reset(self):
        pass

    def detect(self, inputs):
        rgbs = inputs["rgb"].permute(0, 2, 3, 1).cpu().numpy()

        # add target objectgoal to classes
        if "objectgoal" in inputs.keys():
            objgoal = inputs["objectgoal"]
            changeflag = False
            for b in range(len(objgoal)):
                if not objgoal[b] in self.classes:
                    self.classes.append(objgoal[b])
                    changeflag = True
            if changeflag:
                self.change_category(self.classes)

        pred_boxes = []
        scores = []
        classes = []
        pred_masks = []
        for b in range(rgbs.shape[0]):
            rgb = rgbs[0]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            predictions = self.predictor(bgr)
            predictions = predictions["instances"]
            pred_boxes.append(predictions.pred_boxes)
            scores.append(predictions.scores)
            #classes.append(predictions.pred_classes.tolist())
            classes.append(predictions.pred_classes)
            pred_masks.append(predictions.pred_masks)

        max_det = max([pred_boxes[b].tensor.shape[0] for b in range(len(pred_boxes))])
        boxes = -1 * torch.ones(self.batch_size, max_det, 6).to(self.device)
        masks = torch.zeros(self.batch_size, max_det, inputs["rgb"].shape[2], inputs["rgb"].shape[3]).to(self.device)
        for b in range(rgbs.shape[0]):
            if pred_boxes[b].tensor.shape[0] > 0:
                boxes[b, :pred_boxes[b].tensor.shape[0], :4] = pred_boxes[b].tensor
                boxes[b, :pred_boxes[b].tensor.shape[0], 4] = scores[b]
                boxes[b, :pred_boxes[b].tensor.shape[0], 5] = classes[b]
                masks[b, :pred_boxes[b].tensor.shape[0], :, :] = pred_masks[b]
                
        outputs = {
            "objects": {
                "names": self.get_config()["objects"]["names"],
                "boxes": boxes,
                "masks": masks,
            },
        }
        return outputs
        


