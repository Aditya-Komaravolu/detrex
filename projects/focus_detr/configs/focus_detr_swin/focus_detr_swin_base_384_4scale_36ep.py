#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

from detrex.config import get_config
from ..models.focus_detr_swin_base_384 import model

# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("snaglist_train", {}, "/home/aditya/snaglist_dataset_mar11/annotations/train.json", "/home/aditya/snaglist_dataset_mar11/train")
# register_coco_instances("snaglist_val", {}, "/home/aditya/snaglist_dataset_mar11/annotations/valid.json", "/home/aditya/snaglist_dataset_mar11/valid")


# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep
train = get_config("common/train.py").train

# modify training config
# train.init_checkpoint = "/home/aditya/detrex/configs/common/focus_detr_swin_base_384_4scale_22k_36ep.pth"
train.init_checkpoint = "/home/aditya/detrex/focus_detr_swin_base_384_4scale_22k_36ep.pth"
train.output_dir = "/home/aditya/focus_detr_training/output/"

# max training iterations
train.max_iter = 270000

# run evaluation every 5000 iters
train.eval_period = 1000

# log training infomation every 20 iters
train.log_period = 100

# save checkpoint every 5000 iters
train.checkpointer.period = 1000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.dataset.filter_empty = True
dataloader.train.total_batch_size = 16
# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
