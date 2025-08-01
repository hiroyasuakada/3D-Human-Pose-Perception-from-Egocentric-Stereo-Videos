# 3D-Human-Pose-Perception-from-Egocentric-Stereo-Videos (CVPR 2024)

The official PyTorch inference code of our CVPR 2024 paper, **"3D Human Pose Perception from Egocentric Stereo Videos"**.

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

Note that UnrealEgo2 is fully compatible with [UnrealEgo](https://github.com/hiroyasuakada/UnrealEgo). The test data of UnrealEgo is **publicly available**, including 72 body joint annotations (32 for body and 40 for hand), whereas the test data of UnrealEgo2 and UnrealEgo-RW are not (See "Evaluation" section of our benchmark challenge page).

## Depths from SfM/Metashape

You can download depth data from SfM/Metashape described in our paper.

- <a href="https://unrealego.mpi-inf.mpg.de/data/download_unrealego2_test_sfm.sh" download>Depth from UnrealEgo2 test split</a>
- <a href="https://unrealego.mpi-inf.mpg.de/data/download_unrealego_rw_test_sfm.sh" download>Depth from UnrealEgo-RW test split</a>

        bash download_unrealego2_test_sfm.sh
        bash download_unrealego_rw_test_sfm.sh

Note that these depth data differ from the synthetic pixel-perfect depth maps available on [our benchmark challenge page](https://unrealego.mpi-inf.mpg.de/).


## Implementation

### Dependencies

We tested our code with the following dependencies:

- Python 3.9
- Ubuntu 18.04
- PyTorch 2.0.0
- Cuda 11.7

Please install other dependencies:
    
    pip install -r requirements.txt    

### Inference

Our inference code automatically generates submittable zip files of the predictions as described in our benchmark challenge page.

#### Trained models

You can download [our trained models](https://drive.google.com/drive/folders/1NQ08KHKNl3iyrcWzgMlUfCoFlQnn97ve?usp=drive_link). Please save them in `./log/(experiment_name)`.

#### Inference on UnrealEgo2 test dataset

        bash scripts/test/unrealego2_pose-qa-avg-df_data-ue2_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad.sh

            --data_dir [path to the `UnrealEgoData2_test_rgb` dir]
            --metadata_dir [path to the `UnrealEgoData2_test_sfm` dir]

Please modify the arguments above. The pose predictions will be saved in `./results/UnrealEgoData2_test_pose (raw and zip versions)`.

#### Inference on UnrealEgo-RW test dataset

- Model without pre-training on UnrealEgo2
  
        bash scripts/test/unrealego2_pose-qa-avg-df_data-ue-rw_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad.sh

            --data_dir [path to the `UnrealEgoData_rw_test_rgb` dir]
            --metadata_dir [path to the `UnrealEgoData_rw_test_sfm` dir]

- Model with pre-training on UnrealEgo2
  
        bash scripts/test/unrealego2_pose-qa-avg-df_data-ue2_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad_finetuning_epoch5-5.sh

            --data_dir [path to the `UnrealEgoData_rw_test_rgb` dir]
            --metadata_dir [path to the `UnrealEgoData_rw_test_sfm` dir]

Please modify the arguments above. The pose predictions will be saved in `./results/UnrealEgoData_rw_test_pose (raw and zip versions)`.

**For quantitative results of your methods, please follow the instructions in [our benchmark challenge page](https://unrealego.mpi-inf.mpg.de/) and submit a zip version.**

Also, note that UnrealEgo2 is fully compatible with [UnrealEgo](https://github.com/hiroyasuakada/UnrealEgo). This means that you can train your method on UnrealEgo2 and test it on UnrealEgo, and vice versa. The test data of UnrealEgo is **publicly available**, including 72 body joint annotations (32 for body and 40 for hand), whereas the test data of UnrealEgo2 and UnrealEgo-RW are not (See "Evaluation" section of our benchmark challenge page).

