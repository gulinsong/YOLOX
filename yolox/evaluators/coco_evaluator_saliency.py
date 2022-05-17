#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np
import math
import torch
import cv2

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.evaluators import COCOEvaluator

class COCOWithSaliencyEvaluator(COCOEvaluator):
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """
    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        super(COCOEvaluator, self).__init__()
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        data_list_saliency = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                
                # modified
                outputs_saliency, outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            #data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))
            data_list_saliency.extend(self.convert_to_saliency_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            data_list_saliency = gather(data_list_saliency, dst=0)
            data_list_saliency = list(itertools.chain(*data_list_saliency))
            torch.distributed.reduce(statistics, dst=0)

        #eval_results = self.evaluate_prediction(data_list, statistics)
        eval_results_saliency = self.evaluate_saliency_prediction(data_list_saliency, statistics)
        synchronize()
        #return eval_results, eval_results_saliency
        return eval_results_saliency

    def convert_to_saliency_format(self, outputs, info_imgs, ids):
        data_list = []
        for idx, (img_h, img_w, img_id) in enumerate(zip(
            info_imgs[0], info_imgs[1], ids
        )):
            output = outputs[idx].cpu().numpy()

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            output = cv2.resize(output, (self.img_size[0]/scale, self.img_size[1]/scale))
            output = output[:int(img_h), :int(img_w)]

            pred_data = {
                "image_id": int(img_id),
                "category_id": -1,
                "bbox": [],
                "score": [],
                "segmentation": [],
                "saliency_map": output
            }
            data_list.append(pred_data)
        return data_list


    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [], 
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_saliency_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"
        
        cocoGt = self.dataloader.dataset.coco
        gt_pred_pairs = []
        l1_sum = 0.0
        r_simi = 0.0
        r_cc = 0.0
        r_nss = 0.0
        # for each test imgs:
        for img_i in data_dict:
            img_id = img_i['image_id']
            pred_saliency_map = img_i['saliency_map']
            pred_saliency_map = (pred_saliency_map - np.min(pred_saliency_map))/((np.max(pred_saliency_map)-np.min(pred_saliency_map))*1.0)
            gt_saliency_map = np.zeros_like(pred_saliency_map)
            gt_h, gt_w = gt_saliency_map.shape
            ann_ids = cocoGt.getAnnIds(imgIds=img_id)
            targets = cocoGt.loadAnns(ann_ids)
            for target in targets:
                x = int(target['bbox'][0] + target['bbox'][2] / 2)
                y = int(target['bbox'][0] + target['bbox'][2] / 2)
                if x > 0 and x < gt_w and y > 0 and y < gt_h:
                    gt_saliency_map[y, x] = 1 # check
            #l1_sum = np.abs(pred_saliency_map-gt_saliency_map).sum()
            r_simi += self.similarity(pred_saliency_map, gt_saliency_map)
            r_cc += np.abs(self.cc(pred_saliency_map, gt_saliency_map))
            r_nss += self.nss(pred_saliency_map, gt_saliency_map)
        if len(data_dict) > 0:
            info = "simi: {}, cc: {}, nss: {}".format(r_simi/len(data_dict), r_cc/len(data_dict), r_nss/len(data_dict))
            return r_simi/len(data_dict), r_nss/len(data_dict), info
        else:
            info = "simi: {}, cc: {}, nss: {}".format(0, 0, 0)
            return 0, 0, info

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
    
    def similarity(self, s_map,gt):
        # here gt is not discretized nor normalized
        #s_map = normalize_map(s_map)
        #gt = normalize_map(gt)
        s_map = s_map/(np.sum(s_map)*1.0)
        gt = gt/(np.sum(gt)*1.0)
        x,y = np.where(gt>0)
        sim = 0.0
        for i in zip(x,y):
            sim = sim + min(gt[i[0],i[1]],s_map[i[0],i[1]])
        return sim
    
    def nss(self, s_map,gt):
        #gt = discretize_gt(gt)
        s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)

        x,y = np.where(gt==1)
        temp = []
        for i in zip(x,y):
            temp.append(s_map_norm[i[0],i[1]])
        return np.mean(temp)

    def cc(self, s_map,gt):
        s_map_norm = (s_map - np.mean(s_map))/np.std(s_map)
        gt_norm = (gt - np.mean(gt))/np.std(gt)
        a = s_map_norm
        b= gt_norm
        r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
        return r        
