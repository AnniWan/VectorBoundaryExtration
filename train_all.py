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
                        default="config/config.yaml", help="config,file")
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


def  validate(val_loader, model,val_config):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    end = time.time()

    for i, (input, target, field, point_path) in enumerate(val_loader):     
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

        mean_loss = torch.mean(total_loss)
        intersection, union, target = intersectionAndUnionGPU(
            seg_out, target, val_config['classes'])
        intersection, union, target = intersection.cpu(
        ).numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / \
            (sum(target_meter.val) + 1e-10)
        loss_meter.update(mean_loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        logger.info('Test: [{}/{}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                    'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                      data_time=data_time,
                                                      batch_time=batch_time,
                                                      loss_meter=loss_meter,
                                                      accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info(
        'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(val_config['classes']):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i,
                    iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mIoU, mAcc, allAcc

def train(train_loader, model,optimizer, epoch, train_config):
    torch.cuda.empty_cache()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()

    for i, (input, target, field, point_path) in enumerate(train_loader):
        
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

        # if epoch<15:
        #     total_loss = 0.9*(ce_loss + be_loss + bd_loss   + bd2_loss + con_loss)+0.1*point_loss
        # else:
        total_loss = 0.1*(ce_loss + be_loss + bd_loss   + bd2_loss + con_loss)+0.9*point_loss
        total_loss = total_loss.float()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        logger.info('Epoch: [{}/{}][{}/{}]'.format(epoch+1, train_config['epochs'], i + 1,len(train_loader)))
        intersection, union, target = intersectionAndUnionGPU(seg_out.max(1)[1], target, train_config['classes'])
        intersection, union, target = intersection.cpu(
        ).numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(
            union), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / \
            (sum(target_meter.val) + 1e-10)
        

        if (i + 1) % train_config['print_freq']==0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'MainLoss {total_loss:.4f} '
                        'point_loss {point_loss:4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, train_config['epochs'], i + 1, len(train_loader),                                  
                                                          total_loss=total_loss,
                                                          point_loss =point_loss,
                                                          accuracy=accuracy)
                        )
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
        epoch+1, train_config['epochs'], mIoU, mAcc, allAcc))
    
    return total_loss, mIoU, mAcc, allAcc


def main():
    torch.cuda.empty_cache()
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
    optimizer = torch.optim.Adam(model.parameters(),lr=train_config['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25], gamma=0.1, last_epoch=-1, verbose=False)
    if 0:
        net_dict = model.state_dict()
        predict_model = torch.load('pre_model\hrnetv2_w48_imagenet_pretrained.pth')
        state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
        net_dict.update(state_dict)  # 将预训练参数更新到新的网络层
        model.load_state_dict(net_dict)  # 加载预训练参数
    if train_config['weights']:
        if os.path.isfile(train_config['weights']):
            logger.info("=> loading weight '{}'".format(
                train_config['weights']))
            checkpoint = torch.load(train_config['weights'], map_location=lambda storage, loc: storage.cuda())
            start_epoch = checkpoint['epoch']
            mIou_best =  checkpoint['best_mIoU']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])           
    else:
        logger.info("=> no weight found at '{}'".format(
            train_config['weights']))
   
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_transform = transform.Compose([
        transform.Resize([train_h, train_w]),
        transform.Color_jitter(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
        )
    train_data = dataset.PolyData(
        split='train', point_length =train_config['point_length'],data_root=config['Dataset']['data_root'], data_list=config['Dataset']['train_list'], transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=train_config['workers'], pin_memory=True,  drop_last=True)
    if train_config['evaluate']:
        val_transform = transform.Compose([
            transform.Resize([train_h, train_w]),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        val_data = dataset.PolyData(
            split='val', point_length =train_config['point_length'], data_root=config['Dataset']['data_root'], data_list=config['Dataset']['test_list'], transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size_val, shuffle=False, num_workers=train_config['workers'], pin_memory=True)
    IoU =[]
    mIou_best =0.0
    if os.path.exists( 'out/'+train_config['save_path'] ):
        os.makedirs( 'out/'+train_config['save_path'] )
    for epoch in range(start_epoch, epochs):
        epoch_log = epoch+1
        total_loss, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader, model, optimizer,epoch, train_config)
        scheduler.step()
        if (epoch_log % train_config['save_freq'] == 0):
            filename = 'out/'+train_config['save_path'] +  '/train_epoch_' + str(epoch_log) + '.pth'
        logger.info('Saving checkpoint to: ' + filename)
              
        torch.save({
                'epoch': epoch_log,
                'best_mIoU': mIoU_train,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
            }, filename)

        
    if train_config['evaluate'] :
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, train_config)               
            IoU.append([loss_val, mIoU_val, mAcc_val, allAcc_val])

            if mIou_best < mIoU_val:
                final_output_dir =train_config['save_path'] + \
                    '/best_model' + '.pth'
                mIou_best = mIoU_val
                torch.save({
                    'epoch': epoch_log,
                    'best_mIoU': mIoU_train,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),
                }, final_output_dir)
               
            

    # with open(os.path.join(train_config['save_path'],"log.txt"),'w') as v:
    #     for o in IoU:
    #         v.write(str(o)+"\n")

if __name__ == '__main__':
    main()