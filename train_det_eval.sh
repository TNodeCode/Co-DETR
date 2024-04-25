export CONFIG_DIR=configs

export DATASET_DIR=data/spine
export CLASSES="classes.txt"
export ANNOTATIONS_TRAIN=$DATASET_DIR/annotations/instances_train2017.json
export ANNOTATIONS_VAL=$DATASET_DIR/annotations/instances_val2017.json
export ANNOTATIONS_TEST=$DATASET_DIR/annotations/instances_test2017.json
export IMAGES_TRAIN=$DATASET_DIR/train2017
export IMAGES_VAL=$DATASET_DIR/val2017
export IMAGES_TEST=$DATASET_DIR/test2017

export MODEL_TYPE=faster_rcnn
export MODEL_NAME=faster_rcnn_x101_64x4d_fpn_1x_coco
export BATCH_SIZE=8
export EPOCHS=25
export WORK_DIR=runs/$MODEL_TYPE/$MODEL_NAME

# Train the model
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

# Detect bounding boxes for all datasets and epochs
for DATASET in "train" "val" "test"
do
    for EPOCH in $(seq 1 $EPOCHS)
    do
        python cli.py detect \
            --model_type $MODEL_TYPE \
            --model_name $MODEL_NAME \
            --weight_file epoch_$EPOCH.pth \
            --image_files $DATASET_DIR/$DATASET"'2017/*.png'" \
            --results_file detections_${DATASET}_epoch_${EPOCH}.csv \
            --batch_size $BATCH_SIZE \
            --device cuda:0
    done
done

# Evaluate detected bounding boxes for training dataset
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_TRAIN \
    --epochs $EPOCHS \
    --csv_file_pattern detections_train_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_train.csv

# Evaluate detected bounding boxes for validation dataset
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_VAL \
    --epochs $EPOCHS \
    --csv_file_pattern detections_val_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_val.csv

# Evaluate detected bounding boxes for test dataset
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_TEST \
    --epochs $EPOCHS \
    --csv_file_pattern detections_test_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_test.csv

