# AFPS (Neural Networks)
The official implementation of "Weakly supervised temporal action localization with actionness-guided false positive suppression".

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



