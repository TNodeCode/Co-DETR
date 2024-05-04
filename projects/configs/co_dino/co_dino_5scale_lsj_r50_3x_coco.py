import config

_base_ = [
    'co_dino_5scale_lsj_r50_1x_coco.py'
]
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=config.get_number_of_epochs(36))