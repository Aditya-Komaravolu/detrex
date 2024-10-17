from functools import partial
from detrex.config import get_config
from detrex.modeling.backbone.eva import get_vit_lr_decay_rate
import detectron2.data.transforms as T
from detectron2 import model_zoo
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)

import torch

from detectron2.solver.build import get_default_optimizer_params
from detectron2.evaluation import COCOEvaluator
# from detectron2.data.datasets import register_coco_instances

from ..models.dino_eva_02 import model
# from ..common.coco_loader_lsj_1280 import dataloader





# get default config
# optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
# train = get_config("common/train.py").train



SGD = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=0.02,
    momentum=0.9,
    weight_decay=1e-4,
)


AdamW = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

optimizer = AdamW




# Common training-related configs that are designed for "tools/train_net.py"
# You can use your own instead, together with your own train_net.py
train = dict(
    # Directory where output files are written to
    output_dir="./output",
    # The initialize checkpoint to be loaded
    init_checkpoint="",
    # The total training iterations
    max_iter=90000,
    # options for Automatic Mixed Precision
    amp=dict(enabled=True),
    # options for DistributedDataParallel
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=True,
    ),
    # options for Gradient Clipping during training
    clip_grad=dict(
        enabled=False,
        params=dict(
            max_norm=0.1,
            norm_type=2,
        ),
    ),
    # training seed
    seed=-1,
    # options for Fast Debugging
    fast_dev_run=dict(enabled=False),
    # options for PeriodicCheckpointer, which saves a model checkpoint
    # after every `checkpointer.period` iterations,
    # and only `checkpointer.max_to_keep` number of checkpoint will be kept.
    checkpointer=dict(period=5000, max_to_keep=100),
    # run evaluation after every `eval_period` number of iterations
    eval_period=5000,
    # output log to console every `log_period` number of iterations.
    log_period=20,
    # logging training info to Wandb
    # note that you should add wandb writer in `train_net.py``
    wandb=dict(
        enabled=False,
        params=dict(
            dir="./wandb_output",
            project="detrex",
            name="detrex_experiment",
        )
    ),
    # model ema
    model_ema=dict(
        enabled=False,
        decay=0.999,
        device="",
        use_ema_weights_for_eval_only=False,
    ),
    # the training device, choose from {"cuda", "cpu"}
    device="cuda",
    # ...
)


train = L.load(train)
optimizer = L.load(AdamW)


# modify model config
model.backbone.net.img_size = 1280 
model.backbone.square_pad = 1280  
model.backbone.net.patch_size = 16  
model.backbone.net.window_size = 16  
model.backbone.net.embed_dim = 1024
model.backbone.net.depth = 24
model.backbone.net.num_heads = 16
model.backbone.net.mlp_ratio = 4*2/3
model.backbone.net.use_act_checkpoint = True
model.backbone.net.drop_path_rate = 0.4  

# 5, 11, 17, 23 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
)

# modify training config
train.init_checkpoint = "/home/aditya/detrex/dino_eva_02_m38m_pretrain_vitdet_l_4attn_1280_lrd0p8_4scale_12ep.pth"
train.output_dir = "/home/aditya/dino-eva_training"

# max training iterations
train.max_iter = 90000


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
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.8, num_layers=24)
optimizer.params.overrides = {}
optimizer.params.weight_decay_norm = None




dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="snaglist_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            L(T.RandomFlip)(horizontal=True),
        ],
        image_format="BGR",
        use_instance_mask=True,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="snaglist_val)", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)



# Data using LSJ
image_size = 1280
# dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 64
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]



# modify dataloader config
dataloader.train.num_workers = 4

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4

