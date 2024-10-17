#please change the dataset registry  and output folder path in `projects/dino/train_net.py` if you are using any new dataset 
python3 tools/train_net.py \
    --config-file projects/dino/configs/dino-internimage/dino_internimage_large_4scale_12ep.py \
    --num-gpus 1 