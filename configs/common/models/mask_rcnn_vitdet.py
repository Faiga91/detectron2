from functools import partial
import torch
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from .mask_rcnn_fpn import model
from ..data.constants import constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"

# Base
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

#model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"
model.roi_heads.box_head.conv_norm = "LN"
model.roi_heads.mask_in_features = []
model.roi_heads.mask_head = None

model.roi_heads.num_classes = 1

model.roi_heads.box_head.fc_dims = [512]


box2box_transform = Box2BoxTransform(weights=(10, 10, 5, 5))

model.roi_heads.box_predictor = FastRCNNOutputLayers(
    input_shape=512, 
    num_classes=model.roi_heads.num_classes,
    box2box_transform=box2box_transform,
    test_score_thresh=0.05, 
    test_nms_thresh=0.5,
)

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [512]
