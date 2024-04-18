export CONFIG_DIR=projects/configs/co_dino
export MODEL=co_dino_5scale_swin_large_16e_o365tococo_spine
export EPOCH=25

export DATASET_DIR=C:/Users/tilof/PycharmProjects/DeepLearningProjects/MasterThesis/CVDataInspector/datasets/spine

python test.py $CONFIG_DIR/$MODEL.py "./runs/$MODEL/epoch_$EPOCH.pth" --eval bbox