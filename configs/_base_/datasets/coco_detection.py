import os

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/spine/' if not os.getenv("DATASET_DIR") else os.getenv("DATASET_DIR") + "/"
classes = ['spine']
train_ann_file=data_root + 'annotations/instances_train2017.json' if not os.getenv("DATASET_TRAIN_ANNOTATION") else os.getenv("DATASET_TRAIN_ANNOTATION")
train_img_prefix=data_root + 'train2017/' if not os.getenv("DATASET_TRAIN_IMAGES") else os.getenv("DATASET_TRAIN_IMAGES")
val_ann_file=data_root + 'annotations/instances_val2017.json' if not os.getenv("DATASET_VAL_ANNOTATION") else os.getenv("DATASET_VAL_ANNOTATION")
val_img_prefix=data_root + 'val2017/' if not os.getenv("DATASET_VAL_IMAGES") else os.getenv("DATASET_VAL_IMAGES")
test_ann_file=data_root + 'annotations/instances_test2017.json' if not os.getenv("DATASET_TEST_ANNOTATION") else os.getenv("DATASET_TEST_ANNOTATION")
test_img_prefix=data_root + 'test2017/' if not os.getenv("DATASET_TEST_IMAGES") else os.getenv("DATASET_TEST_IMAGES")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
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
