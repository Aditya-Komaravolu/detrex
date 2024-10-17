FOLDER_NAME=$1
CONF=$2

python3 demo.py \
    --config-file /home/aditya/detrex/projects/maskdino/configs/maskdino_r50_coco_instance_seg_50ep.py \
    --input /home/aditya/${FOLDER_NAME}/extracted_frames_200/*.jpg \
    --output /home/aditya/${FOLDER_NAME}_images_maskdino_r50_coco_instance_seg_50ep_model_0007999_conf_${CONF} \
    --confidence-threshold 0.5 \
    --min_size_test 720 \
    --max_size_test 1280 \
    --metadata_dataset "snaglist_train" \
    --opts train.init_checkpoint=/home/aditya/maskdino_r50_coco_instance_seg_50ep_training_d4_may9/model_0007999.pth