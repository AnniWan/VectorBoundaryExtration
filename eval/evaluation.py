import argparse

from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from metrics.polis import PolisEval
from metrics.angle_eval import ContourEval
from metrics.cIoU import compute_IoU_cIoU

def coco_eval(annFile, resFile,ids):
    type=1
    annType = ['bbox', 'segm']
    print('Running demo for *%s* results.' % (annType[type]))

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt, cocoDt, annType[type])
    cocoEval.params.imgIds = ids
    cocoEval.params.catIds = [100]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats

def boundary_eval(annFile, resFile,ids):
    dilation_ratio = 0.02 # default settings 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = BCOCOeval(cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio)
    cocoEval.params.imgIds = ids
    cocoEval.evaluate()
    
    cocoEval.accumulate()
    cocoEval.summarize()

def polis_eval(annFile, resFile,ids):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco,ids)

    polisEval.evaluate()

def max_angle_error_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    contour_eval = ContourEval(gt_coco, dt_coco)
   
    max_angle_diffs = contour_eval.evaluate()
    print('Mean max tangent angle error(MTA): ', max_angle_diffs.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="res.json")
    parser.add_argument("--dt-file", default="test.annotation.poly.simple.tol_1.json")
    parser.add_argument("--eval-type", default="ciou", choices=["coco_iou", "boundary_iou", "polis", "angle", "ciou"])
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file
    import json
    data =json.load(open(dt_file,'r'))
    ids=[754,852,5850,8706,13643,14183,15446,15797,17614,18983]
    ooo = []
    for d in data:
         if d['image_id'] in ids:
             ooo.append(d)
    # for ann in data:
    #     ids.append(ann['image_id'])
    if eval_type == 'coco_iou':
        coco_eval(gt_file, ooo,ids)
    elif eval_type == 'boundary_iou':
        boundary_eval(gt_file, ooo,ids)
    elif eval_type == 'polis':
        polis_eval(gt_file, ooo,ids)
    elif eval_type == 'angle':
        max_angle_error_eval(gt_file, ooo)
    elif eval_type == 'ciou':
        compute_IoU_cIoU( gt_file,ooo)
    else:
        raise RuntimeError('please choose a correct type from \
                            ["coco_iou", "boundary_iou", "polis", "angle", "ciou"]')
