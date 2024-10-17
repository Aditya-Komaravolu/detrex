#please change the dataset registry  and output folder path in `projects/dino/train_net.py` if you are using any new dataset 
python3 tools/train_net.py \
    --config-file projects/dino/configs/dino-vitdet/dino_vitdet_large_4scale_50ep.py \
    --num-gpus 2 