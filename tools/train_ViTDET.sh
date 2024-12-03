#!/bin/bash
python lazyconfig_train_net.py \
--config ../projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py \
> logs/ViTDet_train_ous_copy.txt 2>&1