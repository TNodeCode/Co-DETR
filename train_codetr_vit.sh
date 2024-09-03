python train.py \
    projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py \
    --cfg-options \
    dataset_type=CocoSpineDataset \
    data.samples_per_gpu=1 \
    data.workeers_per_gpu=1 \
    data.train.type=CocoSpineDataset \
    data.val.type=CocoSpineDataset \
    data.test.type=CocoSpineDataset \
    model.query_head.num_classes=1 \
    model.roi_head.0.bbox_head.num_classes=1 \
    model.bbox_head.0.num_classes=1 \