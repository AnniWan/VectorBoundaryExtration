from ast import parse
import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import json 
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import yaml
import seg.hrnet_all
import datasets.transform as transform
import datasets.dataset as dataset
import datasets.loss as loss
import seg.gcn as gcn
import shapely
from metrics.eval import *
import metrics.coco

def get_parser():
    parser = argparse.ArgumentParser(description="ploygon")
    parser.add_argument("--config", type=str,
                        default="config/test.yaml", help="config,file")
    args = parser.parse_args()
    assert args.config is not None
    f = open(args.config, 'r', encoding='utf-8')
    cont = f.read()
    config = yaml.load(cont)
    return config


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0]/2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max()-X.min(), Y.max()-Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox

def single_annotation(image_id, poly,id):
    _result = {}
    _result["image_id"] = image_id
    _result["category_id"] = 100 
    _result["score"] = 1
    _result["segmentation"] = [poly]
    bbox= bounding_box_from_points(_result["segmentation"])
    _result["bbox"] = bbox
    _result["area"] = bbox[-1]*bbox[-2]
    _result["id"] = id
    return _result

class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def  validate(val_loader, model, val_config,out_save_name):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    
    model.eval()
    end = time.time()
    max_ta=[]
    annotations=[]
    ids =[]
    for i, (input, target, field, point_path,name) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_point = []
        for path in point_path:
            data = json.load(open(path ))
        
            shapes= data["shapes"]
            points =[]
            for shape in shapes:
                point =shape["points"]
                contours = np.array(point).astype('int')
                poly = dataset.point_resize(contours,1,1)
                
                points.append(poly)
            batch_point.append(points)
        field = [field[0].cuda(non_blocking=True),field[1].cuda(non_blocking=True)]

        out,corners,out_points,ce_loss, be_loss,bd_loss,bd2_loss,con_loss,point_loss = model(input,target,field,batch_point)
   
        seg_out = out[:, :-2, :, :]
        total_loss = 0.1*(ce_loss + be_loss + bd_loss   + bd2_loss + con_loss)+0.9*point_loss
        total_loss = total_loss.float()
        
        if val_config["save_polygon"]:
            if os.path.exists( './res/'+out_save_name+'/input/' ):
                os.makedirs('./res/'+out_save_name+'/input/' )
            backimage = cv2.imread(os.path.join("F:/dataset/crowAI/val/images",name[0]))          
            aa =[]
            for point in out_points[0]:
                dp_points = cv2.approxPolyDP(point, 3, True).reshape([-1,2])
                aa.append(dp_points)
            cv2.drawContours(backimage,aa,-1,(255,255,154),2)            
            for point in aa:
                for x,y in point :
                    cv2.circle(backimage,(x,y),1,(102,0,255),2) 
            cv2.imwrite('./res/'+out_save_name+'/input1/'+name[0], backimage,[int(cv2.IMWRITE_JPEG_QUALITY), 100]+[int(cv2.IMWRITE_JPEG_OPTIMIZE), 0])

        annotation = metrics.coco.seg_coco(seg_out,name[0],aa,is_points=True)   
        ids.append(int(os.path.splitext(name[0])[0])) 
        annotations.extend(annotation)
        n = input.size(0)

        mean_loss = torch.mean(total_loss)
        
        if val_config["save_mask"]:
            if os.path.exists( './res/'+out_save_name+'/mask/' ):
                os.makedirs('./res/'+out_save_name+'/mask/' )
            output = seg_out.max(1)[1]
            sample_pred =output.data.cpu().numpy()
            cv2.imwrite('./res/'+out_save_name+'/mask/'+name[0], sample_pred[0]*255,[int(cv2.IMWRITE_JPEG_QUALITY), 100]+[int(cv2.IMWRITE_JPEG_OPTIMIZE), 0])
        
        loss_meter.update(mean_loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        logger.info('Test: [{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    .format(i + 1, len(val_loader),
                                                      data_time=data_time,
                                                      batch_time=batch_time,
                                                      loss_meter=loss_meter))
    if val_config["save_prediction_json"]:
        file ="./output_json/"+"predictions_"+out_save_name+"_.json"
        with open(file, "w") as fp:
            fp.write(json.dumps(annotations))
        fp.close()
    states = metrics.coco.eval_one(file,ids)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg,states


def main():
    config = get_parser()
    train_config = config['Train']
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    start_epoch = train_config['start_epoch']
    batch_size_val = train_config['batch_size_val']
    train_h, train_w = train_config['train_h'], train_config['train_w']
    cont = open("config/HRnet.yaml",'r',encoding='utf-8').read()
    net_config = yaml.load(cont)
    global logger
    logger = get_logger()
    logger.info(train_config)
    logger.info("=> creating model ...")
    model = seg.hrnet_all.HighResolutionNet(net_config).cuda()
    IoU =[]
    mIou_best =0.0
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    
    test_transform = transform.Compose([
                transform.Resize([train_h, train_w]),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
    test_data = dataset.PolyData(
                split='test', point_length =train_config['point_length'],data_root=config['Dataset']['data_root'], data_list=config['Dataset']['test_list'], transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=batch_size, shuffle=False, num_workers=train_config['workers'], pin_memory=True)
   
    if  os.path.exists(train_config['weights']) :           
            if train_config['weights']:
                if os.path.isfile(train_config['weights']):
                    logger.info("=> loading weight '{}'".format(
                        train_config['weights']))
                    checkpoint = torch.load(train_config['weights'])
                    model.load_state_dict(checkpoint['state_dict'])  

                    out_save_name = train_config["weights"].split('\\')[1]   
                    print(out_save_name)  
                    loss_val,stats = validate(test_loader, model, train_config,out_save_name) 
                    fp = open("out/"+"out_save_name"+'/coco.txt', 'a',encoding='utf-8')           
                    fp.write(str(stats))  
                    fp.write('\n')                      
                    fp.close()               
          
            else:
                logger.info("=> no weight found at '{}'".format(
                    train_config['weights']))
 


if __name__ == '__main__':
    main()
