python3 demo.py \
    --config-file /home/aditya/detrex/projects/deta/configs/deta_r50_5scale_12ep.py \
    --input /home/aditya/vartwrfl11/extracted_frames_200/*.jpg \
    --output /home/aditya/vartwrfl11_deta_pred \
    --confidence-threshold 0.3 \
    --min_size_test 720 \
    --max_size_test 1280 \
    --metadata_dataset "snaglist_train" \
    --opts train.init_checkpoint=/home/aditya/deta_res50_backbone_training/model_0031999.pth 
