python3 demo.py \
    --config-file /home/aditya/detrex/projects/dino/configs/dino-vitdet/dino_vitdet_large_4scale_50ep.py \
    --input /home/aditya/floor2_8july_frames/*.jpg \
    --output /home/aditya/floor2_8july_frames_dino_vit_pred_sep9_model_0022499 \
    --confidence-threshold 0.4 \
    --min_size_test 1080 \
    --max_size_test 1920 \
    --metadata_dataset "snaglist_train" \
    --opts train.init_checkpoint=/home/aditya/dino_vitdet_large_50ep_training_sep9/best_model1.pth