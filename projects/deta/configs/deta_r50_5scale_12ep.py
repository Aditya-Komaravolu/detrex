from detrex.config import get_config
from .models.deta_r50 import model
from .scheduler.coco_scheduler import lr_multiplier_12ep_10drop as lr_multiplier

# using the default optimizer and dataloader
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
# train.init_checkpoint = "/home/aditya/detrex/converted_deta_swin_o365_finetune.pth"
train.init_checkpoint = "/home/aditya/detrex/converted_deta_r50_5scale_12ep.pth"   #pretained_checkpoint
# train.init_checkpoint = "/home/aditya/deta_res50_backbone_training/model_0031999.pth"    #our finetuned model for predicitons
train.output_dir = "/home/aditya/deta_res50_backbone_training"

# max training iterations
train.max_iter = 90000
train.eval_period = 1000
train.checkpointer.period = 1000

# set training devices
train.device = "cuda"
model.device = train.device

# modify dataloader config
dataloader.train.num_workers = 8

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 8

