export DATASET=train
export EPOCH=2
python cli.py detect \
    --model_type faster_rcnn \
    --model_name faster_rcnn_r50_fpn_1x_coco \
    --weight_file epoch_$EPOCH.pth \
    --image_files data/spine/$DATASET"'2017/*.png'" \
    --results_file detections_${DATASET}_epoch_${EPOCH}.csv \
    --batch_size 16 \
    --device cuda:0