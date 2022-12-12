import os
import fnmatch

import shapely.geometry
from tqdm import tqdm
from multiprocess import Pool
import json

# COCO:
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import  pycocotools
from pycocotools.cocoeval import Params
import datetime
import time
from collections import defaultdict
import copy
from functools import partial
import numpy as np
import skimage
import skimage.io
import skimage.morphology
import skimage.measure
from skimage import measure,color
import matplotlib.pyplot as plt
import cv2
def seg_coco(seg,id,points,is_points=False):
    annotations = []
    # Have to convert binary mask to a list of annotations
    seg = seg[0].max(0)[1]
    seg = seg.data.cpu().numpy()
    seg_mask = seg
    if is_points:
        for point in points:           
            point= point.reshape([-1,2])
            point = point[point.sum(axis=1)!=0,:]
            x = point[:,0]
            y =  point[:,1]
            xmin,xmax = np.min(x),np.max(x)
            ymin,ymax = np.min(y),np.max(y)
            obj_mask = seg_mask[ymin:ymax,xmin:xmax] 
            bbox = [float(xmin),float(ymin),float(xmax-xmin),float(ymax-ymin)]

            score = float(np.mean(obj_mask))
            image_id = int(os.path.splitext(id)[0])
            area = cv2.contourArea(point)
            if area < 20:
                continue
            annotation = {
                "category_id": 100,  # Building
                "bbox": bbox,
                "segmentation":[point.reshape([-1]).astype('float').tolist()],
                "score": 1.0,
                "area" : area,
                "image_id": image_id}
            annotations.append(annotation)
    else:
    # for i, contour_props in enumerate(properties):
    #     skimage_bbox = contour_props["bbox"]
    #     obj_mask = seg_mask[skimage_bbox[0]:skimage_bbox[2], skimage_bbox[1]:skimage_bbox[3]]    
    #     score = float(np.mean(obj_mask))
    #     coco_bbox = [skimage_bbox[1], skimage_bbox[0],
    #                  skimage_bbox[3] - skimage_bbox[1], skimage_bbox[2] - skimage_bbox[0]]
    #     image_mask = labels == (i + 1)  # The mask has to span the whole image
    #     
    #     rle = pycocotools.mask.encode(np.asfortranarray(image_mask))
    #     rle["counts"] = rle["counts"].decode("utf-8")
    #     image_id = int(os.path.splitext(id)[0])
    #     annotation = {
    #         "category_id": 100,  # Building
    #         "bbox": coco_bbox,
    #         "segmentation": rle,
    #         "score": score,
    #         "area" : contour_props["area"],
    #         "id"  : name,
    #         "image_id": image_id}
    #     annotations.append(annotation)
        props = skimage.measure.regionprops(skimage.morphology.label(seg > 0.50))
        for prop in props:
            if ((prop.bbox[2] - prop.bbox[0]) > 0) & ((prop.bbox[3] - prop.bbox[1]) > 0):
                prop_mask = np.zeros_like(seg, dtype=np.uint8)
                prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
                
                image_id = int(os.path.splitext(id)[0])
                masked_instance = np.ma.masked_array(seg, mask=(prop_mask != 1))
                score = masked_instance.mean()
                encoded_region = pycocotools.mask.encode(np.asfortranarray(prop_mask))
                ann_per_building = {
                    'image_id': image_id,
                    'category_id': 100,
                    'segmentation': {
                        "size": encoded_region["size"],
                        "counts": encoded_region["counts"].decode("utf-8")
                    },
                    'score': float(score),
                }
                annotations.append(ann_per_building)
        # labels = skimage.morphology.label(seg)
        
        # properties = skimage.measure.regionprops(labels, cache=True)
        # for i, contour_props in enumerate(properties):
        #     skimage_bbox = contour_props["bbox"]
        #     obj_mask = seg[skimage_bbox[0]:skimage_bbox[2], skimage_bbox[1]:skimage_bbox[3]]   
        
        # # score = float(np.mean(obj_mask))
        #     coco_bbox = [skimage_bbox[1], skimage_bbox[0],
        #                 skimage_bbox[3] - skimage_bbox[1], skimage_bbox[2] - skimage_bbox[0]]

        #     image_mask = labels == (i + 1)  # The mask has to span the whole image
        #     rle = pycocotools.mask.encode(np.asfortranarray(image_mask))
        #     rle["counts"] = rle["counts"].decode("utf-8")
        #     image_id = int(os.path.splitext(id)[0])
        #     area =(skimage_bbox[3] - skimage_bbox[1])*( skimage_bbox[2] - skimage_bbox[0])
        #     annotation = {
        #         "category_id": 100,  # Building
        #         "bbox": coco_bbox,
        #         "segmentation": rle,
        #         "score": 1.0,
        #         "id"  : name,
        #         "area" :area,
        #         "image_id": image_id}
        #     annotations.append(annotation)
    return annotations 

def eval_one(file,ids):
    gt_annotation_filename = "F:/dataset/crowAI/val/annotation-small.json"
    cocoGt = COCO(gt_annotation_filename)
    cocodt = cocoGt.loadRes(file)
    cocoeval = COCOeval(cocoGt,cocodt,'segm')
    cocoeval.params.imgIds = ids
    cocoeval.params.catIds = [100]
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()

        # Save stats
    stats = {}
    stat_names = ["AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L", "AR", "AR_50", "AR_75", "AR_S", "AR_M", "AR_L"]
    assert len(stat_names) == cocoeval.stats.shape[0]
    for i, stat_name in enumerate(stat_names):
            stats[stat_name] = cocoeval.stats[i]
    print(stats)
    return stats


