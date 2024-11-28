from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.coco_loader_lsj import dataloader


NUM_CLASSES = 1

register_coco_instances("my_custom_dataset_train", {}, "/dataset/hyper-kvasir/train-COCO-annotations.json", "/dataset/hyper-kvasir/train")
register_coco_instances("my_custom_dataset_val", {}, "/dataset/hyper-kvasir/val-COCO-annotations.json", "/dataset/hyper-kvasir/val")

dataloader.train.dataset.names = "my_custom_dataset_train"
dataloader.test.dataset.names = "my_custom_dataset_val"

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.roi_heads.num_classes = NUM_CLASSES
model.roi_heads.mask_head = None 
model.roi_heads.mask_in_features = None
model.roi_heads.mask_pooler = None

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
#train.max_iter = 184375
train.max_iter  = 1000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        #milestones=[163889, 177546],
        milestones=[500, 800],
        num_updates=train.max_iter,
    ),
    warmup_length=2 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
