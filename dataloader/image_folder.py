import os, glob
import os.path
from natsort import natsorted


def make_dataset(opt, data_list_path, data_sub_path, motion_id=None, mode=None):

    list_img_paths = []

    num_frame_total = (opt.seq_len - 1) * opt.num_frame_skip + 1

    with open(data_list_path) as f:
        paths = [s.strip() for s in f.readlines()]

    count_total_frames = 0
    count_motion = 0

    for path in paths:

        full_path = os.path.join(path, data_sub_path, "*.png")

        list_imgs_per_sequence = natsorted(glob.glob(full_path))

        if len(list_imgs_per_sequence) == 0:
            print("No file found: {}".format(full_path))
            continue

        count_motion += 1


        if (opt.seq_len > 1):
            num_seed_imgs = (opt.seq_len - 1) * opt.num_frame_skip

            first_frame = list_imgs_per_sequence[0]
            multi_first_frames = [first_frame for i in range(num_seed_imgs)]
            list_imgs_per_sequence = multi_first_frames + list_imgs_per_sequence

        len_list_imgs_per_sequence = len(list_imgs_per_sequence)

        # Make dataset for sigle-frame-based methods
        if opt.seq_len == 0:
            list_img_paths += list_imgs_per_sequence

        # Make dataset for video-based methods
        else:
            if len_list_imgs_per_sequence < num_frame_total:
                continue
            else:
                for i in range(0, len_list_imgs_per_sequence - num_frame_total + 1):
                    list_img_paths += [list_imgs_per_sequence[i: i + num_frame_total: opt.num_frame_skip]]

    print("count_motion: {}".format(count_motion))

    return list_img_paths, len(list_img_paths)


def make_sfm_depth_dataset(opt, video_data_paths, num_data, camera):

    list_depth_data_paths = []
    list_depth_mask_data_paths = []

    for i in range(num_data):

        list_depth_data_paths_i = []
        list_depth_mask_data_paths_i = []

        video_data_path = video_data_paths[i]

        # try to get depth data corresponding to the frame data
        for frame_data_path in video_data_path:
            frame_data_path_id = int(os.path.basename(frame_data_path).replace("final_", "").replace(".png", ""))

            start_id = (frame_data_path_id // 100) * 100
            remainder_id = (frame_data_path_id - start_id) % 3
            depth_subdir_name = "start" + str(start_id) + "_end" + str(start_id + 99) + "_remainder" + str(remainder_id)

            depth_data_path = frame_data_path.replace(opt.data_dir, opt.metadata_dir).replace("/fisheye_final_image/{}/final_{}.png".format(camera, frame_data_path_id), "/{}/sfm_depth/{}/{}/depth_{}.png".format(opt.depth_dir_name, depth_subdir_name, camera, frame_data_path_id))

            if os.path.exists(depth_data_path):
                list_depth_data_paths_i.append(depth_data_path)
                list_depth_mask_data_paths_i.append(depth_data_path.replace("/sfm_depth", "/sfm_depth_mask").replace("/depth_", "/mask_"))
            else:
                list_depth_data_paths_i.append("NONE")
                list_depth_mask_data_paths_i.append("NONE")

        list_depth_data_paths += [list_depth_data_paths_i]
        list_depth_mask_data_paths += [list_depth_mask_data_paths_i]

    return list_depth_data_paths, list_depth_mask_data_paths

