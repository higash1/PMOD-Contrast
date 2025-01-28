import torch.nn.functional as F
import torch
from torch import Tensor

from typing import List, Dict, Any, Tuple
import random

class VisualizeFeatureMap:
    def __init__(self, device):
        self.d_features:Tensor = torch.zeros(0).to(device)
        self.s_features:Tensor = torch.zeros(0).to(device)

        self.d_label_imgs = torch.zeros((0, 3, 20, 20)).to(device)
        self.s_label_imgs = torch.zeros((0, 3, 20, 20)).to(device)

        self.d_labels: List[int] = []
        self.s_labels: List[int] = []
        
        self.features: Tensor = torch.zeros(0).to(device)
        self.label_imgs = torch.zeros((0, 3, 20, 20)).to(device)
        self.labels: List[int] = []
        
        red_image: Tensor = torch.zeros((3, 20, 20)).to(device)
        red_image[0, :, :] = 1
        self.red_image: Tensor = red_image
        
        blue_image: Tensor = torch.zeros((3, 20, 20)).to(device) 
        blue_image[2, :, :] = 1
        self.blue_image: Tensor = blue_image
    
    def extract_features(self, features_vector: Tensor, gt_label: Tensor):
        """
        gt_label: ground truth label (b, c, h, w)
        features_vector: feature map (b, c, h', w')
        """
        resize_shape:Tuple = features_vector.size()
        gt_lab = gt_label.float()
        # resize
        resize_gt_label: Tensor = F.interpolate(gt_lab, size=(resize_shape[2], resize_shape[3]), mode='nearest').int()
        init_cond: Tensor = torch.logical_or(resize_gt_label == 3, resize_gt_label == 4)[0]
        
        vis_features_vector: Tensor = features_vector[0, :, :, :].clone()

        # dynamic vs static
        if torch.sum(init_cond) > 0:
            
            dynamic_channels: Tuple = torch.where(init_cond)[0]
            static_channels: Tuple = torch.where(~init_cond)[0]

            dynamic_bin: Tensor = vis_features_vector[random.choice(dynamic_channels.tolist()), :, :]
            static_bin: Tensor = vis_features_vector[random.choice(static_channels.tolist()), :, :]
            
            self.features = torch.cat((self.features, dynamic_bin.view(1, -1)), dim=0)
            self.features = torch.cat((self.features, static_bin.view(1, -1)), dim=0)
            self.label_imgs = torch.cat((self.label_imgs, self.red_image.view(1, 3, 20, 20)), dim=0)
            self.label_imgs = torch.cat((self.label_imgs, self.blue_image.view(1, 3, 20, 20)), dim=0)
            self.labels.append(1)
            self.labels.append(0)

        # only static
        else:
            static_channels: Tuple = torch.where(~init_cond)[0]
            static_bin: Tensor = vis_features_vector[random.choice(static_channels.tolist()), :, :]
            self.features = torch.cat((self.features, static_bin.view(1, -1)), dim=0)
            self.label_imgs = torch.cat((self.label_imgs, self.blue_image.view(1, 3, 20, 20)), dim=0)
            self.labels.append(0)
            
    def get_features(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "label_imgs": self.label_imgs,
            "labels": self.labels
        }
            
    def reset_embeddings(self):
        self.features = torch.zeros(0).to(self.features.device)
        self.label_imgs = torch.zeros((0, 3, 20, 20)).to(self.label_imgs.device)
        self.labels = []
    