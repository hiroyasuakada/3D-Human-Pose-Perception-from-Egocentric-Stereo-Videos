
# script
python predict_pose.py \
    --project_name UnrealEgoPose \
    --experiment_name unrealego2_pose-qa-avg-df_data-ue2_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad_finetuning_epoch5-5 \
    --depth_dir_name unrealego_heatmap_shared_ue2_B16_epoch5-5_finetuning_epoch1-1 \
    --model unrealego2_pose_qa_df \
\
    --use_slurm \
    --use_amp \
    --init_ImageNet \
    --compile \
    --batch_size 1 \
    --seq_len 5 \
    --num_frame_skip 3 \
\
    --data_dir /CT/UnrealEgo/static00/UnrealEgoData_realworld_test_rgb \
    --metadata_dir /CT/UnrealEgo/static00/UnrealEgoData_realworld_metadata \
