#please change the dataset registry  and output folder path in `projects/dino/train_net.py` if you are using any new dataset 
python3 tools/train_net.py \
    --config-file projects/maskdino/configs/maskdino_r50_coco_instance_seg_50ep.py \
    --num-gpus 1 