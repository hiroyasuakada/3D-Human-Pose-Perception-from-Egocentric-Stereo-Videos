# 3D-Human-Pose-Perception-from-Egocentric-Stereo-Videos (CVPR 2024)

Official PyTorch inference code of our CVPR 2024 paper, **"3D Human Pose Perception from Egocentric Stereo Videos"**.

![img](doc/overview_setup.png)

**For any questions, please contact the first author, Hiroyasu Akada [hakada@mpi-inf.mpg.de] .**

[[Project Page](https://4dqv.mpi-inf.mpg.de/UnrealEgo2/)] [[Benchmark Challenge](https://unrealego.mpi-inf.mpg.de/)]

## Citation

```
    @inproceedings{hakada2024unrealego2,
      title = {3D Human Pose Perception from Egocentric Stereo Videos},
      author = {Akada, Hiroyasu and Wang, Jian and Golyanik, Vladislav and Theobalt, Christian},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year = {2024}
    }
```



## UnrealEgo2/UnrealEgo-RW Datasets

### Download

You can download the **UnrealEgo2/UnrealEgo-RW datasets** on [our benchmark challenge page](https://unrealego.mpi-inf.mpg.de/).


## Depths from SfM/Metashape

You can donwload depth data from SfM/Metashape described in our paper.

- <a href="https://unrealego.mpi-inf.mpg.de/data/download_unrealego2_test_sfm.sh" download>Depth from UnrealEgo2 test split</a>
- <a href="https://unrealego.mpi-inf.mpg.de/data/download_unrealego_rw_test_sfm.sh" download>Depth from UnrealEgo-RW test split</a>

        bash download_unrealego2_test_sfm.sh
        bash download_unrealego_rw_test_sfm.sh

Note that these depth data are different from the synthetic depth maps available on [our benchmark challenge page](https://unrealego.mpi-inf.mpg.de/).


## Implementation

### Dependencies

We tested our code with the following dependencies:

- Python 3.9
- Ubuntu 18.04
- PyTorch 2.0.0
- Cuda 11.7

### Inference

#### Inference on UnrealEgo2 test dataset

        bash scripts/test/unrealego2_pose/unrealego2_pose-qa-avg-df_data-ue2_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad.sh

The pose predictions will be saved in `./results/UnrealEgoData2_test_pose (raw and zip versions)`.

#### Inference on UnrealEgo-RW test dataset

        # Without fine-tuning
        bash scripts/test/unrealego2_pose/unrealego2_pose-qa-avg-df_data-ue-rw_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad.sh

        # With fine-tuning
        bash scripts/test/unrealego2_pose_finetuning/unrealego2_pose-qa-avg-df_data-ue2_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad_finetuning_epoch5-5.sh

The pose predictions will be saved in `./results/UnrealEgoData_rw_test_pose (raw and zip versions)`.

**For quantitative results of your methods, please follow the instructions in [our benchmark challenge page](https://unrealego.mpi-inf.mpg.de/) and submit a zip version.**

