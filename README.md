# STAS_Object_Detection

## About

T-brain AI實戰吧 - 肺腺癌病理切片影像之腫瘤氣道擴散偵測競賽 I：運用物體偵測作法於找尋STAS

## Prerequisites

***TWCC Container:***
> - pytorch-21.06-py3:latest

## Installation

Step 0. Git clone the project folder.
```
git clone https://github.com/kuokai0429/STAS_Object_Detection.git
cd STAS_Object_Detection
```

Step 1. Create pipenv environment under current project folder and Install project dependencies.
```
pip3 install pipenv
pipenv --python 3.8
pipenv shell
pipenv install --skip-lock
```

Step 2. Clone and Install MMDetection under current project folder.
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python setup.py develop
```

Step 3. Install additional tools for mmcv and mmdetection.
```
cd ..
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

Step 4. Download Configurations, Pretrained Weights and Datasets from shared Google Drive 
> ***Source:*** <br> https://drive.google.com/file/d/1xBELX0HR1kkloxPZc-m_rjxWKAZNqITP/view?usp=sharing

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xBELX0HR1kkloxPZc-m_rjxWKAZNqITP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xBELX0HR1kkloxPZc-m_rjxWKAZNqITP" -O STAS_OBJ_Data.zip && rm -rf /tmp/cookies.txt
```

Step 5. Move /Configs, /Weights and /Datasets from /STAS_OBJ_Data to current project folder.
```
rm -r Configs Weights Datasets
unzip STAS_OBJ_Data.zip
mv STAS_OBJ_Data/Configs .
mv STAS_OBJ_Data/Weights .
mv STAS_OBJ_Data/Datasets .
rmdir STAS_OBJ_Data
```

## Inference

1. Detecting COCO Objects on demo.jpg from mmdetection.
```
python demo_coco.py --weight_file mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.pth
```

2. Detecting STAS on Public and Private image Dataset.
```
python demo_stas_obj.py --weight_dir htc_swin_s_7_088_Weights --testdata_dirs Public_Image Private_Image/Image
```

## Training

```
python train_stas_obj.py --weight_dir htc_swin_s_7_088_Weights --config_file htc_without_semantic_swin_fpn.py
```
