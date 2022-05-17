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
from scipy import ndimage
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

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SaliencyEvaluator(COCOEvaluator):
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
        losses = AverageMeter()
        auc = AverageMeter()
        aae = AverageMeter()

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

        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
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

            data_list_saliency.extend(self.evaluate_each_batch(outputs_saliency, targets, info_imgs))

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


    def evaluate_each_batch(self, outputs, targets, info_imgs):
        data_list = []
        for idx, (img_h, img_w) in enumerate(zip(
            info_imgs[0], info_imgs[1]
        )):
            output = outputs[idx].cpu().numpy().squeeze()
            target = targets[idx].cpu().numpy().squeeze()
            aae1, auc1, _ = self.computeAAEAUC(output,target)

            # preprocessing: resize
            #scale = min(
            #    self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            #)
            #output = cv2.resize(output, (self.img_size[0]/scale, self.img_size[1]/scale))
            #output = output[:int(img_h), :int(img_w)]

            res_data = {
                "image_id": int(-1),
                "category_id": -1,
                "bbox": [],
                "score": [],
                "segmentation": [],
                "aae": aae1,
                "auc":auc1
            }
            data_list.append(res_data)
        return data_list

    def evaluate_saliency_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

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
        l1_sum = 0.0
        r_simi = 0.0
        r_cc = 0.0
        r_nss = 0.0
        auc = AverageMeter()
        aae = AverageMeter()
        # for each test imgs:
        for img_i in data_dict:
            auc.update(img_i['auc'])
            aae.update(img_i['aae'])

        if len(data_dict) > 0:
            info = "auc: {}, aae: {}".format(auc.avg, aae.avg)
            return auc.avg, aae.avg, info
        else:
            info = "auc: {}, aae: {}".format(0, 0)
            return 0, 0, info

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

    def computeAAEAUC(self, output, target):
        aae = []
        auc = []
        gp = []
        if output.ndim == 3:
            for batch in range(output.shape[0]):
                out_sq = output[batch,:,:].squeeze()
                tar_sq = target[batch,:,:].squeeze()
                h, w = out_sq.shape
                r = int(h/2)
                predicted = ndimage.measurements.center_of_mass(out_sq)
                (i,j) = np.unravel_index(tar_sq.argmax(), tar_sq.shape)
                gp.append([i,j])
                d = r/math.tan(math.pi/6)
                r1 = np.array([predicted[0]-r, predicted[1]-r, d])
                r2 = np.array([i-r, j-r, d])
                angle = math.atan2(np.linalg.norm(np.cross(r1,r2)), np.dot(r1,r2))
                aae.append(math.degrees(angle))

                z = np.zeros((h,w))
                z[int(predicted[0])][int(predicted[1])] = 1
                z = ndimage.filters.gaussian_filter(z, 14)
                z = z - np.min(z)
                z = z / np.max(z)
                atgt = z[i][j]
                fpbool = z > atgt
                auc1 = 1 - float(fpbool.sum())/output.shape[2]/output.shape[1]
                auc.append(auc1)
            return np.mean(aae), np.mean(auc), gp
        else:
            h, w = output.shape
            r = int(h/2)
            predicted = ndimage.measurements.center_of_mass(output)
            (i,j) = np.unravel_index(target.argmax(), target.shape)
            d = r/math.tan(math.pi/6)
            r1 = np.array([predicted[0]-r, predicted[1]-r, d])
            r2 = np.array([i-r, j-r, d])
            angle = math.atan2(np.linalg.norm(np.cross(r1,r2)), np.dot(r1,r2))
            aae = math.degrees(angle)

            z = np.zeros((h,w))
            z[int(predicted[0])][int(predicted[1])] = 1
            z = ndimage.filters.gaussian_filter(z, 14)
            z = z - np.min(z)
            z = z / np.max(z)
            atgt = z[i][j]
            fpbool = z > atgt
            auc = (1 - float(fpbool.sum())/(output.shape[0]*output.shape[1]))
            return aae, auc, [[i,j]]       
