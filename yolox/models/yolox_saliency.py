#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOXSaliency(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head_det=None, head_saliency=None):
        super().__init__()
        #if backbone is None:
        #    backbone = YOLOPAFPN()
        #if head is None:
        #    head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head_det
        self.head_saliency = head_saliency

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss_saliency = self.head_saliency(fpn_outs, targets, x)
            #loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {
                "total_loss": loss_saliency,
                "iou_loss": 0,#iou_loss,
                "l1_loss": 0,#l1_loss,
                "conf_loss": 0,#conf_loss,
                "cls_loss": 0,#cls_loss,
                "num_fg": 0,#num_fg,
            }
        else:
            outputs_saliency = self.head_saliency(fpn_outs)
            outputs_det = self.head(fpn_outs)
            outputs = [outputs_saliency, outputs_det]
        return outputs
