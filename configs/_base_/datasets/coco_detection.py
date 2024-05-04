import os
import config

# dataset settings
dataset_type = 'CocoDataset'
classes = config.get_classes()
train_ann_file=config.get_train_annotation_file('data/annotations/instances_train2017.json')
train_img_prefix=config.get_train_image_dir('data/train2017/')
val_ann_file=config.get_val_annotation_file('data/annotations/instances_val2017.json')
val_img_prefix=config.get_val_image_dir('data/val2017/')
test_ann_file=config.get_test_annotation_file('data/annotations/instances_test2017.json')
test_img_prefix=config.get_test_image_dir('data/test2017/')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    *config.get_augmentations(),
    #dict(type='MixUp', img_scale=(640, 640), ratio_range=(0.5, 1.5), flip_ratio=0.5),
    #dict(type='Mosaic', img_scale=(640, 640), center_ratio_range=(0.5, 1.5)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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
data = dict(
    samples_per_gpu=config.get_batch_size(2),
    workers_per_gpu=config.get_workers_per_gpu(2),
    train=dict(
        type=dataset_type,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        pipeline=train_pipeline,
        classes = classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        pipeline=test_pipeline,
        classes = classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=test_ann_file,
        img_prefix=test_img_prefix,
        pipeline=test_pipeline,
        classes = classes,
    )
)
evaluation = dict(interval=1, metric='bbox')
