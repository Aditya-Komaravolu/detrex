from detrex.config import get_config
from ..models.dino_internimage import model

# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "/home/aditya/detrex/dino_internimage_large_4scale_12ep.pth"
train.output_dir = "/home/aditya/dino_internimage_large_384_4scale_12ep_training_may13"

# max training iterations
train.max_iter = 200000
train.eval_period = 1000
train.log_period = 100
train.checkpointer.period = 1000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"


model.device = train.device
model.num_classes = 2

# modify optimizer config
optimizer.lr = 0.5e-5
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 3e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 4

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4