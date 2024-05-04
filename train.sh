export CONFIG_DIR=configs
export DATASET_DIR=data/spine
export ANNOTATIONS_TRAIN=$DATASET_DIR/annotations/instances_train2017.json
export ANNOTATIONS_VAL=$DATASET_DIR/annotations/instances_val2017.json
export ANNOTATIONS_TEST=$DATASET_DIR/annotations/instances_test2017.json
export IMAGES_TRAIN=$DATASET_DIR/train2017
export IMAGES_VAL=$DATASET_DIR/val2017
export IMAGES_TEST=$DATASET_DIR/test2017
export CLASSES="classes.txt"
export WORKERS=4
export BATCH_SIZE=16
export MODEL_TYPE=faster_rcnn
export MODEL_NAME=faster_rcnn_r50_fpn_1x_coco
export EPOCHS=25
export WORK_DIR=runs/$MODEL_TYPE/$MODEL_NAME

python cli.py train \
    --config_dir $CONFIG_DIR \
    --train_annotations $ANNOTATIONS_TRAIN \
    --train_images $IMAGES_TRAIN \
    --val_annotations $ANNOTATIONS_VAL \
    --val_images $IMAGES_VAL \
    --test_annotations $ANNOTATIONS_TEST \
    --test_images $IMAGES_TEST \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --classes classes.txt \
    --batch_size $BATCH_SIZE \
    --work_dir $WORK_DIR