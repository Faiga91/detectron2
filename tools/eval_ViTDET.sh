#!/bin/bash
python lazyconfig_train_net.py \
--config ../projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py \
--eval-only \
train.init_checkpoint=output/model_final.pth \
> logs/ViTDet_eval.txt 2>&1