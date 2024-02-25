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


## Implementation

### Dependencies 

- Python 3.9
- Ubuntu 18.04
- PyTorch 1.6.0
- Cuda 10.1

### Inference

#### Inference on UnrealEgo2 test dataset

        bash scripts/test/unrealego2_pose/unrealego2_pose-qa-avg-df_data-ue2_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad.sh

#### Inference on UnrealEgo-RW test dataset

        # Without fine-tuning
        bash scripts/test/unrealego2_pose/unrealego2_pose-qa-avg-df_data-ue-rw_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad.sh

        # With fine-tuning        
        bash scripts/test/unrealego2_pose_finetuning/unrealego2_pose-qa-avg-df_data-ue2_seq5_skip3_B32_lr2-4_pred-seq_local-device_pad_finetuning_epoch5-5.sh


## License Terms
Permission is hereby granted, free of charge, to any person or company obtaining a copy of this dataset and associated documentation files (the "Dataset") from the copyright holders to use the Dataset for any non-commercial purpose. Redistribution and (re)selling of the Dataset, of modifications, extensions, and derivates of it, and of other dataset containing portions of the licensed Dataset, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee.

Packaging or distributing parts or whole of the provided software (including code and data) as is or as part of other datasets is prohibited. Commercial use of parts or whole of the provided dataset (including code and data) is strictly prohibited.

THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE DATASET.
