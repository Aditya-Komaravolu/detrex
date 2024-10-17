FOLDER_NAME=$1
CONF=$2
# FRAMES=$3
python3 demo.py \
    --config-file /home/aditya/detrex/projects/dino/configs/dino-internimage/dino_internimage_large_4scale_12ep.py \
    --input /home/aditya/${FOLDER_NAME}/*.jpg \
    --output /home/aditya/${FOLDER_NAME}_images_dino_internimage_large_4scale_12ep_model_0010999_conf_${CONF} \
    --confidence-threshold ${CONF} \
    --min_size_test 1280 \
    --max_size_test 1280 \
    --metadata_dataset "snaglist_train" \
    --opts train.init_checkpoint=/home/aditya/dino_internimage_large_384_4scale_12ep_training_may13/model_0010999.pth 
