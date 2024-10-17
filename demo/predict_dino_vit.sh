python3 demo.py \
    --config-file /home/aditya/detrex/projects/dino/configs/dino-vitdet/dino_vitdet_large_4scale_50ep.py \
    --input /home/aditya/vartwrfl11/extracted_frames_200/*.jpg \
    --output /home/aditya/vartwrfl11_images_dino_vit_pred_mar24_best_model \
    --confidence-threshold 0.4 \
    --min_size_test 720 \
    --max_size_test 1280 \
    --metadata_dataset "snaglist_train" \
    --opts train.init_checkpoint=/home/aditya/dino_vitdet_large_50ep_training/best_model.pth
