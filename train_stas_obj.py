''' Code for training - Hybrid Task Cascade with Swin Transformer (Swin-S)
    Command: python train_stas_obj.py --weight_dir htc_swin_s_7_088_Weights --config_file htc_without_semantic_swin_fpn.py
'''

print("Running ....")

import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets import build_dataset
from mmdet.apis import set_random_seed
from mmdet.models import build_detector
from mmdet.apis import train_detector

import os
import os.path as osp
from pathlib import Path
import shutil
import argparse

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--weight_dir', required=True, type=str, help="Pretrained weights directory to load. Please put under /Weights folder")
parser.add_argument('--config_file', required=True, type=str, help="Configuration file to load. Please put under /Configs folder")
args = parser.parse_args()


####### Loading directories path

current_dir_root = os.getcwd()
weight_dir_root = args.weight_dir
config_file_root = args.config_file
# weight_dir_root = "htc_swin_s_7_088_Weights"
# config_file_root = "htc_without_semantic_swin_fpn.py"


####### Prepare model config

# Create work directory
dirpath = Path(current_dir_root + "/mmdetection/runs")

if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
    
os.mkdir(current_dir_root + "/mmdetection/runs")

# Prepare the train/valid coco json annotation file
shutil.copytree(current_dir_root + '/Weights/' + weight_dir_root + '/labelme2coco', current_dir_root + "/mmdetection/runs/labelme2coco")

# The new config inherits a base config to highlight the necessary modification
shutil.copy(current_dir_root + '/Configs/' + config_file_root, current_dir_root + '/mmdetection/configs/_base_/models/' + config_file_root)
cfg = Config.fromfile(current_dir_root + '/mmdetection/configs/_base_/models/' + config_file_root)

# Set up working dir to save files and logs.
cfg.work_dir = current_dir_root + '/mmdetection/runs'

# Fixing Issue: " 'ConfigDict' object has no attribute 'device' "
cfg.device = 'cuda' 

# Modify num classes of the model in box head
cfg.model.roi_head.bbox_head[0].num_classes = 1
cfg.model.roi_head.bbox_head[1].num_classes = 1
cfg.model.roi_head.bbox_head[2].num_classes = 1
cfg.model.roi_head.mask_head[0].num_classes = 1
cfg.model.roi_head.mask_head[1].num_classes = 1
cfg.model.roi_head.mask_head[2].num_classes = 1

# Modify dataset related settings ( Augmentation strategy originates from HTC )
cfg.dataset_type = 'CocoDataset'
cfg.classes = current_dir_root + '/mmdetection/runs/labelme2coco/labels.txt'
cfg.data_root = current_dir_root + '/mmdetection/runs/labelme2coco'

cfg.data.test.type = 'CocoDataset'
cfg.data.test.classes = current_dir_root + '/mmdetection/runs/labelme2coco/labels.txt'
cfg.data.test.data_root = current_dir_root + '/mmdetection/runs/labelme2coco'
cfg.data.test.ann_file = current_dir_root + '/mmdetection/runs/labelme2coco/val.json'
cfg.data.test.img_prefix = current_dir_root + '/Datasets/STAS_Train_Datasets'

cfg.data.train.type = 'CocoDataset'
cfg.data.train.classes = current_dir_root + '/mmdetection/runs/labelme2coco/labels.txt'
cfg.data.train.data_root = current_dir_root + '/mmdetection/runs/labelme2coco'
cfg.data.train.ann_file = current_dir_root + '/mmdetection/runs/labelme2coco/train.json'
cfg.data.train.img_prefix = current_dir_root + '/Datasets/STAS_Train_Datasets'

cfg.data.val.type = 'CocoDataset'
cfg.data.val.classes = current_dir_root + '/mmdetection/runs/labelme2coco/labels.txt'
cfg.data.val.data_root = current_dir_root + '/mmdetection/runs/labelme2coco'
cfg.data.val.ann_file = current_dir_root + '/mmdetection/runs/labelme2coco/val.json'
cfg.data.val.img_prefix = current_dir_root + '/Datasets/STAS_Train_Datasets'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=[(400, 1600), (900, 1600)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(900, 1600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

cfg.train_pipeline = train_pipeline
cfg.val_pipeline = test_pipeline
cfg.test_pipeline = test_pipeline

cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.val.pipeline = cfg.val_pipeline
cfg.data.test.pipeline = cfg.test_pipeline

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can also use tensorboard to log the training process
cfg.log_config.interval = 10
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# Modify learning rate config
cfg.lr_config = dict(
    policy='CosineAnnealing', 
    by_epoch=False,
    warmup='linear', 
    warmup_iters= 1000, 
    warmup_ratio= 1.0/10,
    min_lr=1e-07)

# Modify evaluation related settings
cfg.evaluation.interval = 10
cfg.evaluation.save_best = 'auto'

# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 10

meta = dict()
meta['config'] = cfg.pretty_text


####### Train the model

total_training_epochs = 30
batch_size = 4

# Total training epochs
cfg.runner.max_epochs = total_training_epochs

# Batch size
cfg.data.samples_per_gpu = batch_size
cfg.data.workers_per_gpu = 1

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector "without loading checkpoints"
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Train model
train_detector(model, datasets, cfg, distributed=False, validate=True, meta=meta)



