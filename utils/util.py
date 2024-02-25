import os
import numpy as np
import torch


# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8, is_depth=False, is_heatmap=False, is_video=False):
    if image_tensor.dim() == 3:  # (C, H, W)
        image_tensor = image_tensor.cpu().float()
    else: # (B, C, H, W) -> (C, H, W)
        image_tensor = image_tensor[0].cpu().float()

    # size (S, C, H, W) -> (C, H, W) of the last frame of the sequence
    if is_video:
        image_tensor = image_tensor[-1]

    if is_depth:
        image_tensor = image_tensor * bytes
    elif is_heatmap:
        image_tensor = torch.clamp(torch.sum(image_tensor, dim=0, keepdim=True), min=0.0, max=1.0) * bytes
    else:
        # image_tensor = (image_tensor + 1.0) / 2.0 * bytes
        image_tensor = denormalize_ImageNet(image_tensor) * bytes

    image_numpy = (image_tensor.permute(1, 2, 0)).numpy().astype(imtype)
    return image_numpy


def tensor2pose(joints_3d, is_video=False):
    # size (B, S, num of heatmaps, 3) -> (S, num of heatmaps, 3)
    joints_3d = joints_3d[0].cpu().float()

    # size (S, num of heatmaps, 3) -> (num of heatmaps, 3) of the last frame of the sequence
    if is_video:
        joints_3d = joints_3d[-1]

    return joints_3d

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)