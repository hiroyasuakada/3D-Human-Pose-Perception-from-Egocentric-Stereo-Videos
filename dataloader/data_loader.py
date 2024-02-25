import os, glob, sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from dataloader.image_folder import make_dataset, make_sfm_depth_dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        assert (isinstance(image, np.ndarray))
        image -= self.mean
        image /= self.std

        return image


def dataloader(opt, mode='test', motion_id=None):

    dataset = CreateStereoVideoFullDataset(opt, mode, motion_id=motion_id)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
        drop_last=False
    )
    return dataloader

def process_img(transform, img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize([256, 256], Image.BICUBIC)
    img = transform(img)
    img = img.float().numpy()
    return img


def process_depth(transform, depth_path):
    depth = Image.open(depth_path)
    depth = depth.resize([64, 64], Image.BICUBIC)
    depth = transform(depth)
    depth = depth.float().numpy()
    return depth


class CreateStereoVideoFullDataset(torch.utils.data.Dataset):
    def __init__(self, opt, mode="test", motion_id=None):
        super(CreateStereoVideoFullDataset, self).__init__()
        self.opt = opt

        self.type_local_pose = self.opt.type_local_pose
        self.load_size_rgb = opt.load_size_rgb
        self.load_size_heatmap = opt.load_size_heatmap

        self.data_list_path = os.path.join(opt.data_dir, mode + ".txt")

        self.video_data_left_paths, self.num_data_left = make_dataset(
            opt=opt,
            data_list_path=self.data_list_path,
            data_sub_path='fisheye_final_image/camera_left',
            motion_id=motion_id,
            mode=mode,
        )

        self.video_data_right_paths, self.num_data_right = make_dataset(
            opt=opt,
            data_list_path=self.data_list_path,
            data_sub_path='fisheye_final_image/camera_right',
            motion_id=motion_id,
            mode=mode,
        )

        self.depth_video_data_left_paths, self.depth_mask_video_data_left_paths = make_sfm_depth_dataset(
            opt=opt,
            video_data_paths=self.video_data_left_paths,
            num_data=self.num_data_left,
            camera="camera_left"
        )

        self.depth_video_data_right_paths, self.depth_mask_video_data_right_paths = make_sfm_depth_dataset(
            opt=opt,
            video_data_paths=self.video_data_right_paths,
            num_data=self.num_data_right,
            camera="camera_right"
        )

        # transform list: RGB image
        transforms_list = []
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        self.transforms = transforms.Compose(transforms_list)

        # transform list: depth
        transforms_depth_list = []
        transforms_depth_list.append(transforms.ToTensor())
        transforms_depth_list.append(transforms.Normalize((0.5, ), (0.5)))
        self.transforms_depth = transforms.Compose(transforms_depth_list)

        # load fisheye mask
        if "_realworld" in opt.data_dir:
            print("\nWill use UnrealEgo-RW camera mask !!!\n")
            fisheye_edge_mask_path = "./dataloader/fisheye_edge_mask_872_872.png"
            mask_zero_path = "./dataloader/mask_zero_872_872.png"
        else:
            print("\nWill use UnrealEgo 1 or 2 camera mask !!!\n")
            fisheye_edge_mask_path = "./dataloader/fisheye_edge_mask_1024_1024.png"
            mask_zero_path = "./dataloader/mask_zero_1024_1024.png"

        self.fisheye_edge_mask = process_depth(self.transforms_depth, fisheye_edge_mask_path)
        self.mask_zero = process_depth(self.transforms_depth, mask_zero_path)

        self.label_None = np.array(0)
        self.label_Depth = np.array(1)

    def __getitem__(self, index):

        input_video_rgb_left = []
        input_video_rgb_right = []

        sfm_depth_left = []
        sfm_depth_mask_left = []
        sfm_depth_right = []
        sfm_depth_mask_right = []

        np_labels = []

        # get paths for each data
        video_data_left_path = self.video_data_left_paths[index]
        video_data_right_path = self.video_data_right_paths[index]

        depth_video_data_left_path = self.depth_video_data_left_paths[index]
        depth_video_data_right_path = self.depth_video_data_right_paths[index]

        depth_mask_video_data_left_path = self.depth_mask_video_data_left_paths[index]
        depth_mask_video_data_right_path = self.depth_mask_video_data_right_paths[index]

        for i in range(self.opt.seq_len):

            # load RGB image
            frame_data_left_path = video_data_left_path[i]
            frame_data_right_path = video_data_right_path[i]

            frame_data_left = process_img(self.transforms, frame_data_left_path)
            frame_data_right = process_img(self.transforms, frame_data_right_path)

            input_video_rgb_left.append(frame_data_left)
            input_video_rgb_right.append(frame_data_right)

            # load depth image
            depth_data_left_path = depth_video_data_left_path[i]
            depth_data_right_path = depth_video_data_right_path[i]
            depth_mask_data_left_path = depth_mask_video_data_left_path[i]
            depth_mask_data_right_path = depth_mask_video_data_right_path[i]

            if (depth_data_left_path == "NONE") or (depth_data_right_path == "NONE"):
                sfm_depth_left.append(self.fisheye_edge_mask)
                sfm_depth_right.append(self.fisheye_edge_mask)
                sfm_depth_mask_left.append(self.mask_zero)
                sfm_depth_mask_right.append(self.mask_zero)
                np_labels.append(self.label_None)

            else:
                depth_data_left = process_depth(self.transforms_depth, depth_data_left_path)
                depth_data_right = process_depth(self.transforms_depth, depth_data_right_path)
                depth_mask_data_left = process_depth(self.transforms_depth, depth_mask_data_left_path)
                depth_mask_data_right = process_depth(self.transforms_depth, depth_mask_data_right_path)
                sfm_depth_left.append(depth_data_left)
                sfm_depth_right.append(depth_data_right)
                sfm_depth_mask_left.append(depth_mask_data_left)
                sfm_depth_mask_right.append(depth_mask_data_right)
                np_labels.append(self.label_Depth)

        # transform numpy to torch tensor, size B * (L, C, H, W)
        input_video_rgb_left = torch.from_numpy(np.stack(input_video_rgb_left, axis=0)).float()
        input_video_rgb_right = torch.from_numpy(np.stack(input_video_rgb_right, axis=0)).float()
        sfm_depth_left = torch.from_numpy(np.stack(sfm_depth_left, axis=0)).float()
        sfm_depth_mask_left = torch.from_numpy(np.stack(sfm_depth_mask_left, axis=0)).float()
        sfm_depth_right = torch.from_numpy(np.stack(sfm_depth_right, axis=0)).float()
        sfm_depth_mask_right = torch.from_numpy(np.stack(sfm_depth_mask_right, axis=0)).float()
        torch_labels = torch.from_numpy(np.stack(np_labels, axis=0)).float()

        # create dict data
        dict_data = {
            "frame_data_path": video_data_left_path[-1],
            "input_video_rgb_left": input_video_rgb_left,
            "input_video_rgb_right": input_video_rgb_right,
            "sfm_depth_left": sfm_depth_left,
            "sfm_depth_right": sfm_depth_right,
            "sfm_depth_mask_left": sfm_depth_mask_left,
            "sfm_depth_mask_right": sfm_depth_mask_right,
            "torch_labels": torch_labels,
            }

        return dict_data

    def __len__(self):
        return self.num_data_left

