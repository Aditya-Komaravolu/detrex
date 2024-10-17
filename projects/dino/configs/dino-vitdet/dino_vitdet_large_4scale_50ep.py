from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .dino_vitdet_large_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    model,
)


# modify training config
train.max_iter = 375000
train.init_checkpoint = "/home/aditya/detrex/dino_vitdet_large_4scale_50ep.pth"
train.output_dir = "/home/aditya/dino_vitdet_large_50ep_training_sep9"

model.num_classes = 2

model.backbone.net.img_size = 1024
model.backbone.square_pad = 1024
model.backbone.net.embed_dim = 1024


optimizer.lr = 1e-5  # Lower than the previously suggested 5e-5
optimizer.weight_decay = 3e-4  # Up from 1e-4 but less than the earlier 5e-4 suggestion
model.backbone.net.drop_path_rate = 0.4  # Keeps the increase to encourage generalization
train.clip_grad.params.max_norm = 0.6  # Still an option to prevent exploding gradients

dataloader.train.num_workers = 2

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 2

# use warmup lr scheduler
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[300000, 375000],
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)