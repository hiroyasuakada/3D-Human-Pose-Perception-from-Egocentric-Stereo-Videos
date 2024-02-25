#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -c 16
#SBATCH -t 1-23:59:55
#SBATCH -o slurm/output_test_%j.txt
#SBATCH --mem 64G


# conda
source activate py39

# script
python predict_pose.py \
    --project_name UnrealEgoPose \
    --experiment_name unrealego2_pose-qa-avg-df_data-ue2_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad \
    --model unrealego2_pose_qa_df \
    --data_dir /CT/UnrealEgo2/static00/UnrealEgoData2_test_rgb \
    --metadata_dir /CT/UnrealEgo3/static00/UnrealEgoData2_metadata \
    --depth_dir_name unrealego_heatmap_shared_ue2_B16_epoch5-5_1 \
    --path_to_trained_heatmap ./log/unrealego_heatmap_shared_ue2_B16_epoch5-5_1/best_net_HeatMap.pth \
    --camera_left_calib_file /CT/UnrealEgo/work/UnrealEgoPose/main/utils/fisheye/fisheye_calib_unrealego_scaramuzza.json \
    --camera_right_calib_file /CT/UnrealEgo/work/UnrealEgoPose/main/utils/fisheye/fisheye_calib_unrealego_scaramuzza.json \
    --use_slurm \
    --use_amp \
    --init_ImageNet \
    --compile \
    --batch_size 1 \
    --seq_len 5 \
    --num_frame_skip 3 \
    --query_adaptation average \
    --pred_seq_pose \
    --use_depth_padding_mask \

