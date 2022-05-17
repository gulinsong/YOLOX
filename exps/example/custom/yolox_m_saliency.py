#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp
import torch.nn as nn
import torch.distributed as dist
import torch

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/LEDOV"
        self.train_ann = "LEDOV_train.json"
        self.val_ann = "LEDOV_val.json"
        self.data_root_train = "datasets/LEDOV/LEDOV/img_and_heatmap/train_1"
        self.img_list = "datasets/LEDOV/LEDOV/img_and_heatmap/train_imgs1.lst"
        self.gt_list = "datasets/LEDOV/LEDOV/img_and_heatmap/train_gts.lst"
        self.data_root_test = "datasets/LEDOV/LEDOV/img_and_heatmap/val_1"
        self.img_list_test = "datasets/LEDOV/LEDOV/img_and_heatmap/val_imgs1.lst"
        self.gt_list_test = 'datasets/LEDOV/LEDOV/img_and_heatmap/val_gts.lst'
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.cache_img_test = True
        
        self.num_classes = 2

        self.max_epoch = 20
        self.data_num_workers = 4
        self.eval_interval = 1
        self.basic_lr_per_img = 1e-5
        self.scheduler = "multistep"


    def get_model(self):
        from yolox.models import YOLOXSaliency, YOLOPAFPN, YOLOXHead, YOLOXSaliencyHead
        
        from yolox.utils import freeze_module
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
                
        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head_det = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            head_saliency = YOLOXSaliencyHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOXSaliency(backbone, head_det, head_saliency)

        self.model.apply(init_yolo)
        # self.model.head.initialize_biases(1e-2)
        self.model.train()
        freeze_module(self.model.backbone)
        freeze_module(self.model.head)
        return self.model
    
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            COCODataset,
            SaliencyDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = SaliencyDataset(
                data_root=self.data_root_train,
                img_list=self.img_list,
                gt_list=self.gt_list,
                img_size=self.input_size,
                transform=None,
                target_transform=None,
                cache=cache_img,
            )
        '''
        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        '''
        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset, shuffle=True
            )
        else:
            sampler = torch.utils.data.SequentialSampler(self.dataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        #train_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_kwargs)


        batch_sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=False,
        )
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        print('train dataloader: ', len(train_loader))
        return train_loader

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets = nn.functional.interpolate(
                targets, size=tsize, mode="bilinear", align_corners=False
            )
        return inputs, targets

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import SaliencyDataset, ValTransform

        valdataset = SaliencyDataset(
                data_root=self.data_root_test,
                img_list=self.img_list_test,
                gt_list=self.gt_list_test,
                img_size=self.test_size,
                transform=None,
                target_transform=None,
                cache = self.cache_img_test,
            )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader



    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import SaliencyEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = SaliencyEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
