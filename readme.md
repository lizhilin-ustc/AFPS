# AFPS (Neural Networks 2024)
The official implementation of "Weakly supervised temporal action localization with actionness-guided false positive suppression".

## Results
|  Dataset         | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7| AVG(0.1:0.5) | AVG(0.1:0.7) |
| -----------      | --- | --- | ----| ----| ----| ---| -- | ---- | -----|
| THUMOS14         | 73.5| 68.8| 60.8| 51.3| 41.0| 27.5| 16.5| 59.1| 48.5|

|  Dataset         | 0.5 | 0.75 | 0.95 | AVG(0.5:0.95) |
| -----------      | --- | --- | ----| ----|
| ActivityNet 1.2  | 48.6| 29.6| 6.4| 29.9|
| ActivityNet 1.3  | 43.9 |27.1|6.3|27.3|


## Preparation
CUDA Version: 11.3

Pytorch: 1.12.0

Numpy: 1.23.5 

Python: 3.9.7

Dataset: Download the two-stream I3D features for THUMOS'14 to "DATA_PATH". You can download them from [Google Drive](https://drive.google.com/file/d/1paAv3FsqHtNsDO6M78mj7J3WqVf_CgSG/view?usp=sharing).

Update the data_path in "./scripts/train.sh" and "./scripts/inference.sh".


## Training
You can train your own model by executing the following command.
```
    bash ./scripts/train.sh
```


## Inference
You can download our trained model from [here](https://drive.google.com/drive/folders/1-01moeCKpvgZxAiVKDnvAd6mfmQJbBue?usp=drive_link).
Then you need to put the model folder "thumos_AFPS" into the "./outputs" folder.
You can reproduce the results of our experiment by executing the following command.
```
    bash ./scripts/inference.sh
```

## Citation
If this work is helpful for your research, please consider citing our works.
```
@article{li2024weakly,
  title={Weakly supervised temporal action localization with actionness-guided false positive suppression},
  author={Li, Zhilin and Wang, Zilei and Liu, Qinying},
  journal={Neural Networks},
  volume={175},
  pages={106307},
  year={2024},
  publisher={Elsevier}
}
```
