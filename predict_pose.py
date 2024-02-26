import os
import re
import sys
import torch
from collections import defaultdict
import json
import shutil
from options.test_options import TestOptions
from options.base_options import print_opt
from dataloader.data_loader import dataloader
from model.models import create_model

recursive_defaultdict = lambda: defaultdict(recursive_defaultdict)

list_joints_unrealego2 = [
        "head",
        "neck_01",
        "upperarm_l",
        "upperarm_r",
        "lowerarm_l",
        "lowerarm_r",
        "hand_l",
        "hand_r",
        "thigh_l",
        "thigh_r",
        "calf_l",
        "calf_r",
        "foot_l",
        "foot_r",
        "ball_l",
        "ball_r"
    ]

list_joints_unrealego_rw = [
        "Head",
        "Neck",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftFoot",
        "RightFoot",
        "LeftToeBase",
        "RightToeBase",
    ]

def main(opt, model, eval_dataset, epoch, list_joints):

    model.eval()
    bar_eval = enumerate(eval_dataset)

    with torch.no_grad():
        for id, data in bar_eval:
            model.set_input(data)
            model.forward()
            save_pose_json(opt, model, data, list_joints)

    shutil.make_archive(
        "{}/{}".format(opt.save_dir_pose, opt.test_pose_dir_name),
        format="zip",
        root_dir=opt.save_dir_pose,
        base_dir=opt.test_pose_dir_name
    )

def save_pose_json(opt, model, data, list_joints):

    for id in range(opt.batch_size):

        pred_local_pose = model.pred_video_pose[id][-1].cpu().float().numpy()

        dict_data = recursive_defaultdict()

        for joint_id, joint in enumerate(list_joints):
            dict_data[joint] = pred_local_pose[joint_id].tolist()

        frame_data_path = data["frame_data_path"][id]

        save_json_base_name = os.path.basename(frame_data_path).replace("final_", "frame_").replace(".png", ".json")
        save_json_sub_path = re.findall('{}(.*)/fisheye_final_image'.format(opt.data_dir), frame_data_path)[0]

        if save_json_sub_path.startswith("/"):
            save_json_sub_path = save_json_sub_path[1:]

        save_json_path = os.path.join(
            opt.save_dir_pose, # default is "./results"
            opt.test_pose_dir_name,
            save_json_sub_path,
            "json",
            save_json_base_name
        )

        if not os.path.exists(os.path.dirname(save_json_path)):
            os.makedirs(os.path.dirname(save_json_path))

        with open(save_json_path, "w") as m:
            json.dump(dict_data, m, indent=4)


if __name__ == "__main__":

    opt = TestOptions().parse()
    print_opt(opt)

    print("preparing dataset ... ")

    test_dataset = dataloader(opt, mode='test')

    print('test images = {}'.format(len(test_dataset) * opt.batch_size))

    model = create_model(opt)

    opt.test_pose_dir_name = os.path.basename(opt.data_dir).replace("_rgb", "_pose")

    if opt.test_pose_dir_name == "UnrealEgoData2_test_pose":
        list_joints = list_joints_unrealego2
    elif opt.test_pose_dir_name == "UnrealEgoData_rw_test_pose":
        list_joints = list_joints_unrealego_rw
    else:
        raise ValueError('Dataset {} not recognized'.format(opt.test_pose_dir_name))

    print('\n-----------------Predict Pose with Best Model for {}-----------------\n'.format(opt.test_pose_dir_name))

    print("load best model ...")
    model.load_networks("best")
    main(opt, model, test_dataset, "best", list_joints)

    print('\n-----------------All Process Finished-----------------\n')