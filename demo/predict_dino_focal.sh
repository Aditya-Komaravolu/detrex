python3 demo.py \
    --config-file /home/aditya/detrex/projects/dino/configs/dino-focal/dino_focalnet_large_lrf_384_fl4_5scale_12ep.py \
    --input /home/aditya/vtbfl10/extracted_frames_200/*.jpg \
    --output /home/aditya/vtbfl10_dino_focalnet_pred_apr1_d2_model_0007999 \
    --confidence-threshold 0.4 \
    --min_size_test 720 \
    --max_size_test 1280 \
    --metadata_dataset "snaglist_train" \
    --opts train.init_checkpoint=/home/aditya/dino_focalnet_training_apr1_d2/model_0007999.pth
