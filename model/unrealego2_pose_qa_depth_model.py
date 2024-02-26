import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .base_model import BaseModel
from . import network


class UnrealEgo2PoseQADepthModel(BaseModel):
    def name(self):
        return 'UnrealEgo2 Pose Estimator model with Query Adaptation and Depth'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.scaler = GradScaler(enabled=opt.use_amp)

        self.visual_names = [
            'input_video_rgb_left', 'input_video_rgb_right',
            'pred_video_heatmap_left', 'pred_video_heatmap_right',
        ]

        self.visual_pose_names = ["pred_video_pose"]

        self.model_names = ['HeatMap', 'Pose']

        # define the transform network
        self.net_HeatMap = network.define_HeatMap(opt, model=opt.model)
        self.net_Pose = network.define_Pose(opt, model=opt.model)

        self.load_networks(
            net=self.net_HeatMap,
            path_to_trained_weights=opt.path_to_trained_heatmap
            )
        network._freeze(self.net_HeatMap)

        if opt.path_to_trained_pose:
            self.load_networks(
                net=self.net_Pose,
                path_to_trained_weights=opt.path_to_trained_pose
                )

        if opt.compile:
            self.net_HeatMap = torch.compile(self.net_HeatMap)
            self.net_Pose = torch.compile(self.net_Pose)

    def set_input(self, data):
        self.data = data
        self.input_video_rgb_left = data['input_video_rgb_left'].cuda(self.device)
        self.input_video_rgb_right = data['input_video_rgb_right'].cuda(self.device)
        self.sfm_depth_left = data["sfm_depth_left"].cuda(self.device)
        self.sfm_depth_mask_left = data["sfm_depth_mask_left"].cuda(self.device)
        self.sfm_depth_right = data["sfm_depth_right"].cuda(self.device)
        self.sfm_depth_mask_right = data["sfm_depth_mask_right"].cuda(self.device)
        self.torch_labels = data["torch_labels"].cuda(self.device)

    def forward(self):
        with autocast(enabled=self.opt.use_amp):

            ### estimate stereo heatmaps
            with torch.no_grad():
                input_video_rgb_left = rearrange(self.input_video_rgb_left, "b l c h w -> (b l) c h w")
                input_video_rgb_right = rearrange(self.input_video_rgb_right, "b l c h w -> (b l) c h w")

                outputs = self.net_HeatMap(input_video_rgb_left, input_video_rgb_right, return_feat=True)
                pred_video_heatmap_stereo_cat = outputs[0]
                video_feature_stereo = outputs[1]

                # stereo (B*L, 2C, H, W) -> left (B*L, C, H, W) + right (B*L, C, H, W)
                pred_video_heatmap_left, pred_video_heatmap_right = torch.chunk(pred_video_heatmap_stereo_cat, 2, dim=1)
                self.pred_video_heatmap_left = rearrange(pred_video_heatmap_left, "(b l) c h w -> b l c h w", l=self.opt.seq_len)
                self.pred_video_heatmap_right = rearrange(pred_video_heatmap_right, "(b l) c h w -> b l c h w", l=self.opt.seq_len)

            ### estimate pose and reconstruct stereo heatmaps
            sfm_depth_left = rearrange(self.sfm_depth_left, "b l c h w -> (b l) c h w")
            sfm_depth_mask_left = rearrange(self.sfm_depth_mask_left, "b l c h w -> (b l) c h w")
            sfm_depth_right = rearrange(self.sfm_depth_right, "b l c h w -> (b l) c h w")
            sfm_depth_mask_right = rearrange(self.sfm_depth_mask_right, "b l c h w -> (b l) c h w")

            pred_video_pose = self.net_Pose(
                pred_video_heatmap_left,
                pred_video_heatmap_right,
                video_feature_stereo,
                sfm_depth_left,
                sfm_depth_mask_left,
                sfm_depth_right,
                sfm_depth_mask_right,
                self.torch_labels,
            )

            self.pred_video_pose = torch.unsqueeze(pred_video_pose[:, -1], 1)
            self.pred_video_pose_past = pred_video_pose[:, 0:-1]

