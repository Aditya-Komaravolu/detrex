from detectron2.data.datasets import register_coco_instances
import subprocess
register_coco_instances("snaglist_train", {}, "/home/aditya/snaglist_dataset_mar11/annotations/train.json", "/home/aditya/snaglist_dataset_mar11/train")
register_coco_instances("snaglist_val", {}, "/home/aditya/snaglist_dataset_mar11/annotations/valid.json", "/home/aditya/snaglist_dataset_mar11/valid")


process = subprocess.run([
    "python",
    "tools/train_net.py",
    "--config-file",
    "projects/dino_eva/configs/dino-eva-02/dino_eva_02_vitdet_l_4attn_1280_lrd0p8_4scale_12ep.py",
    "--num-gpus",
    "1"
])


