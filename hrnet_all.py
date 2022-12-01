# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mimetypes import init

import os
import logging
import functools

import numpy as np
from sklearn.decomposition import FactorAnalysis

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import yaml
import cv2
import skimage.measure
import skimage.morphology
import shapely.geometry
import shapely.ops
import shapely.prepared
import seg.gcn as gcn
import seg.gan as gan
import networkx as nx
import cv2
import scipy.spatial as spt
import datasets.loss as loss
import math
from shapely.geometry import Polygon
BN_MOMENTUM = 0.1
ALIGN_CORNERS = None
BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
relu_inplace = True
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.size()[-1] != x2.size()[-1]:
            num = x1.size()[-1]-x2.size()[-1] 
  
            pad = nn.ReplicationPad2d(padding=(num,0,0,num))
            x2 = pad(x2)
            
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1
class MultiHead(nn.Module):
    def __init__(self,input_channels,num_class,head_size):
        super(MultiHead,self).__init__()
        m = int(input_channels/4)
        heads=[]
        for output_channel in sum(head_size,[]):
            heads.append(nn.Sequential(
                nn.Conv2d(input_channels,m,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(m,output_channel,kernel_size=1),
            ))
        self.heads=nn.ModuleList(heads)
        assert num_class == sum(sum(head_size,[]))
    
    def forward(self,x):
        return torch.cat([head(x) for head in self.heads],dim=1)
class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        extra = config['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()
        ALIGN_CORNERS = config['MODEL']['ALIGN_CORNERS']

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)


        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=256, 
                kernel_size=extra['FINAL_CONV_KERNEL'],
                stride=1,
                padding=1 if extra['FINAL_CONV_KERNEL']== 3 else 0)
        )

        self.head = MultiHead(256,config['NUM_CLASSES']+2,[[config['NUM_CLASSES']+2]])
        self.contours_conv1 = nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.contours_conv2 = nn.Conv2d(96,128,kernel_size=3,stride=1,padding=1,bias=False)
        self.contours_conv3 = nn.Conv2d(192,256,kernel_size=3,stride=1,padding=1,bias=False)
        self.contours_conv4 = nn.Conv2d(384,512,kernel_size=3,stride=1,padding=1,bias=False)
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
       
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
            )
        self.fc_last = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32, 
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(
                in_channels=32,
                out_channels=2, 
                kernel_size=1,
                stride=1,
                padding=0)
                )    
        if config['GAN']:
            self.gcn = gan.GCN(config['GCN_NUM'],config['GCN_NUM'], alpha = 0.2, n_heads = 2, concat = True)
        else:
            self.gcn = gcn.GCN(config['GCN_NUM'])
        self.mul_num =config['MUL_NUM']
        # self.fc_last = nn.Conv2d(
        #         in_channels=64,
        #         out_channels=2, 
        #         kernel_size=1,
        #         stride=1,
        #         padding=0)
        # self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        # self.fc1 = nn.Linear(in_features=self.stage4_cfg['NUM_CHANNELS'][-1] , out_features=round(self.stage4_cfg['NUM_CHANNELS'][-1] / 16))
        # self.fc2 = nn.Linear(in_features=round(self.stage4_cfg['NUM_CHANNELS'][-1] / 16), out_features=self.stage4_cfg['NUM_CHANNELS'][-1] )
        # self.sigmoid = nn.Sigmoid()

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, input,target,field,batch_point):
        _,c,h,w=input.size()
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.relu(x)
        x = self.layer1(x)

        contour1 = self.contours_conv1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        contour2 = self.contours_conv2(y_list[-1])
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)
        
        contour3 = self.contours_conv3(y_list[-1])
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
 
        x = self.stage4(x_list)    

        contour4 = self.contours_conv4(x[-1])
       
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        d4 = self.decode4(contour4, contour3) # 256,16,16
        
        d3 = self.decode3(d4, contour2) # 256,32,32
        d2 = self.decode2(d3, contour1) # 128,64,64
        d1 = self.decode1(d2) # 64,128,128
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        x = self.head(x)
        
        out = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
        contours =  F.interpolate(self.fc_last(d1), size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)        
        corners = self.corner_detect(out[:,:-2,:,:],out[:,-2:,:,:],contours)
        output = contours.max(1)[1]
        seg_out = out[:, :-2, :, :]
        field_out = out[:, -2:, :, :]
        ce_loss = loss.ce_loss(seg_out, target)
        be_loss = loss.field_loss(field,field_out)
        bd_loss = loss.BD_loss(seg_out, field_out)
        # #ce_loss cal the boundary loss  
        # bd2_loss = loss.BD_loss2(seg_out, contours) 
        # con_loss = loss.ce_loss(contours,target)
        # # mse_loss cal boundary loss
        bd2_loss = loss.BD_loss2(seg_out, contours)
        con_loss = loss.con_loss2(contours,batch_point)
        if self.mul_num:
            point_loss,out_points  = self.multi_batch_cal2(corners,input,seg_out,field_out,batch_point)
        else:
            point_loss,out_points  = self.multi_batch_cal1(corners,seg_out,field_out,batch_point)
        
        return out,corners,out_points,ce_loss, be_loss,bd_loss,bd2_loss,con_loss,point_loss

    def influence(self,input):
        _,c,h,w=input.size()
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.relu(x)
        x = self.layer1(x)

        contour1 = self.contours_conv1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        contour2 = self.contours_conv2(y_list[-1])
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)
        
        contour3 = self.contours_conv3(y_list[-1])
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
 
        x = self.stage4(x_list)    

        contour4 = self.contours_conv4(x[-1])
       
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)

        d4 = self.decode4(contour4, contour3) # 256,16,16
        
        d3 = self.decode3(d4, contour2) # 256,32,32
        d2 = self.decode2(d3, contour1) # 128,64,64
        d1 = self.decode1(d2) # 64,128,128
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        
        out = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)
        contours =  F.interpolate(self.fc_last(d1), size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)        
        corners = self.corner_detect(out[:,:-2,:,:],out[:,-2:,:,:])
        # output = contours.max(1)[1]
        # sample_pred =output.data.cpu().numpy()
        # cv2.imwrite("1.png", sample_pred[0]*255)
        seg_out = out[:, :-2, :, :]
        field_out = out[:, -2:, :, :]      
        out_points  = self.point_influence(corners,seg_out,field_out)
    
        return out,contours ,out_points

    def points_loss(self,point,label):
        n_points = point
        l_points = label
        l_points = l_points[l_points.sum(axis=1)!=0]
        kt = spt.KDTree(l_points, leafsize=10)
        dict_l={}
        for j in range (len(n_points)) :
                #去掉补零数据的影响
                if n_points[j][0] +n_points[j][1] == 0:
                    continue
                d,x = kt.query(n_points[j])
                
                if str(l_points[x]) not in dict_l.keys():
                    dict_l[str(l_points[x])] = [[j,d]]
                else:
                    dict_l[str(l_points[x])].append([j,d])
        res = np.zeros(len(n_points))
        for key,values in dict_l.items():         
                values = sorted(values,key=(lambda x:x[1]))
                res[values[0][0]] = 1
        
        return res

    def cal_iou(self,point,label):
    

        point = point[point.sum(axis=1)!=0,:]
        polygon2 = Polygon(point).buffer(0.001)
        polygon1 = Polygon(label).buffer(0.001)

        inter = polygon1.intersection(polygon2)
        IoU = inter.area
        # px1,py1 = np.min(point[:,0]),np.min(point[:,1])
        # px2,py2 = np.max(point[:,0]),np.max(point[:,1])
    
        # gx1,gy1 = np.min(label[:,0]),np.min(label[:,1])
        # gx2,gy2 = np.max(label[:,0]),np.max(label[:,1])

    
        # parea = (px2 - px1) * (py2 - py1) # 计算P的面积
        # garea = (gx2 - gx1) * (gy2 - gy1) # 计算G的面积

        
        # # 求相交矩形的左上和右下顶点坐标(x1, y1, x2, y2)
        # x1 = max(px1, gx1) # 得到左上顶点的横坐标
        # y1 = min(py1, gy1) # 得到左上顶点的纵坐标
        # x2 = min(px2, gx2) # 得到右下顶点的横坐标
        # y2 = max(py2, gy2) # 得到右下顶点的纵坐标
        
        # # 利用max()方法处理两个矩形没有交集的情况,当没有交集时,w或者h取0,比较巧妙的处理方法
        # # w = max(0, (x2 - x1)) # 相交矩形的长，这里用w来表示
        # # h = max(0, (y1 - y2)) # 相交矩形的宽，这里用h来表示
        # # print("相交矩形的长是：{}，宽是：{}".format(w, h))
        # # 这里也可以考虑引入if判断
        # w = x2 - x1
        # h = y2 - y1
        # if w <=0 or h <= 0:
        #     return 0       
        # area = w * h # G∩P的面积   
        # # 并集的面积 = 两个矩形面积 - 交集面积
        # IoU = area / (parea + garea - area)  
        return IoU
    def cal_features2(self,points,image,seg,filed):
        step = 1
        w,h = seg.size() 
        pad = nn.ZeroPad2d(padding=(2*step, 2*step, 2*step, 2*step))
        pad_seg = pad(seg)
            
        pad_filed = pad(filed)
        pad_images = pad(image)
        G = nx.Graph(name="G")
        # 创建节点
        end =len(points[points.sum(axis=1)!=0])
        edges = []
        for i in range(len(points)):
            G.add_node(i, name=i)
        for i in range(end):
            if not i ==  end-1:
                edge = (i, i+1) 
                edges.append(edge)
            else:
                edge = (i, 0)
                edges.append(edge)               
        # for i in range(len(points)):
        #     if points[i][1]+points[i][0]!=0:
        #         G.add_node(i, name=i)
        #         if not i ==  end-1:
        #             edge = (i, i+1)
        #             edges.append(edge)
        #         else:
        #             edge = (i, 0)
        #             edges.append(edge)
        #     else:
        #         G.add_node(i, name=i)
        # 创建边并添加边到图里
        G.add_edges_from(edges)
        # 从图G获得邻接矩阵（A）和节点特征矩阵（X）
        # nx.draw(G,with_labels=True,font_weight='bold')
        # plt.show()
        A = np.array(nx.attr_matrix(G, node_attr='name')[0])
        X = []
        O = []
        pad_image =  pad_images[2] * 0.299 + pad_images[1] * 0.587 + pad_images[0] * 0.114
        for j in range(len(points)):
            c_x, c_y = points[j]
            x,y = math.floor(c_x)+step,math.floor(c_y)+step
            # x = int(x.data.cpu().numpy())
            # y = int(y.data.cpu().numpy())
            if not j ==  end-1:           
                feature1 = torch.stack([pad_seg[x-step, y-step], pad_seg[x, y-step],pad_seg[x, y+step],
                pad_seg[x-step, y],pad_seg[x, y], pad_seg[x+step, y],pad_seg[x-step, y+step], 
                pad_seg[x, y+step], pad_seg[x+step, y+step],
                pad_image[x-step, y-step], pad_image[x, y-step],pad_image[x, y+step],
                pad_image[x-step, y],pad_image[x, y], pad_image[x+step, y],
                pad_image[x-step, y+step],pad_image[x, y+step], pad_image[x+step, y+step]] ,0) 
                
                feature2=  torch.stack([pad_filed[0][x-step, y-step], pad_filed[0][x, y-step],pad_filed[0][x, y+step],
                pad_filed[0][x-step, y],pad_filed[0][x, y], pad_filed[0][x+step, y],pad_filed[0][x-step, y+step], 
                pad_filed[0][x, y+step], pad_filed[0][x+step, y+step],pad_filed[1][x-step, y-step], pad_filed[1][x, y-step],pad_filed[1][x, y+step],
                pad_filed[1][x-step, y],pad_filed[1][x, y], pad_filed[1][x+step, y],pad_filed[1][x-step, y+step], 
                pad_filed[1][x, y+step], pad_filed[1][x+step, y+step]] ,0)   
                X.append(feature1)
                O.append(feature2)
            else:
 
                X.append(torch.zeros([18]).float().cuda())
                O.append(torch.zeros([18]).float().cuda())
        X = torch.stack(X,0)
        A = torch.from_numpy(np.asarray(A)).float().cuda()
        O = torch.stack(O,0)

        return A, X,O   

    def cal_features1(self,points,seg,filed):
            step = 1
            w,h = seg.size() 
            pad = nn.ZeroPad2d(padding=(2*step, 2*step, 2*step, 2*step))
            pad_seg = pad(seg)
            
            pad_filed = pad(filed)
        
            #pad_filed = pad_filed.data.cpu().numpy()
            G = nx.Graph(name="G")
            # 创建节点
            end =len(points[points.sum(axis=1)!=0])
            edges = []
            for i in range(len(points)):
                G.add_node(i, name=i)
            for i in range(end):
                if not i ==  end-1:
                    edge = (i, i+1) 
                    edges.append(edge)
                else:
                    edge = (i, 0)
                    edges.append(edge)               
            # for i in range(len(points)):
            #     if points[i][1]+points[i][0]!=0:
            #         G.add_node(i, name=i)
            #         if not i ==  end-1:
            #             edge = (i, i+1)
            #             edges.append(edge)
            #         else:
            #             edge = (i, 0)
            #             edges.append(edge)
            #     else:
            #         G.add_node(i, name=i)
            # 创建边并添加边到图里
            G.add_edges_from(edges)
            # 从图G获得邻接矩阵（A）和节点特征矩阵（X）
            # nx.draw(G,with_labels=True,font_weight='bold')
            # plt.show()
            A = np.array(nx.attr_matrix(G, node_attr='name')[0])
            X = []
            O = []
            pad_filed_all = pad_filed[0]+pad_filed[1]
            for j in range(len(points)):
                c_x, c_y = points[j]
                x,y = math.floor(c_x)+step,math.floor(c_y)+step       
                # x = int(x.data.cpu().numpy())
                # y = int(y.data.cpu().numpy())
                if not j ==  end-1:          
                    feature1 = torch.stack([pad_seg[x-step, y-step], pad_seg[x, y-step],pad_seg[x, y+step],
                    pad_seg[x-step, y],pad_seg[x, y], pad_seg[x+step, y],pad_seg[x-step, y+step], 
                    pad_seg[x, y+step], pad_seg[x+step, y+step]] ,0) 
                    feature2=  torch.stack([pad_filed_all[x-step, y-step], pad_filed_all[x, y-step],pad_filed_all[x, y+step],
                    pad_filed_all[x-step, y],pad_filed_all[x, y], pad_filed_all[x+step, y],pad_filed_all[x-step, y+step], 
                    pad_filed_all[x, y+step], pad_filed_all[x+step, y+step]] ,0)  
                    # feature2=  torch.stack([pad_filed[0][x-step, y-step], pad_filed[0][x, y-step],pad_filed[0][x, y+step],
                    # pad_filed[0][x-step, y],pad_filed[0][x, y], pad_filed[0][x+step, y],pad_filed[0][x-step, y+step], 
                    # pad_filed[0][x, y+step], pad_filed[0][x+step, y+step],pad_filed[1][x-step, y-step], pad_filed[1][x, y-step],pad_filed[1][x, y+step],
                    # pad_filed[1][x-step, y],pad_filed[1][x, y], pad_filed[1][x+step, y],pad_filed[1][x-step, y+step], 
                    # pad_filed[1][x, y+step], pad_filed[1][x+step, y+step]] ,0)   
                    X.append(feature1)
                    O.append(feature2)
                else:
    
                    X.append(torch.zeros([9]).float().cuda())
                    O.append(torch.zeros([9]).float().cuda())
            X = torch.stack(X,0)
            A = torch.from_numpy(np.asarray(A)).float().cuda()
            O = torch.stack(O,0)
            return A, X,O

    def multi_batch_cal1(self,batch_points,seg_out,batch_filed,batch_label):
        loss = torch.tensor(0.0)
        points_set = gcn.ChamferLoss2D()
        b,c,w,h =seg_out.size()
        out_points =[]
        
        #im = np.zeros([w,h,3])

        s_loss = 0.0
        for i in range(b):
            points = batch_points[i]
            filed = batch_filed[i]
            label = batch_label[i]
            seg = seg_out[i].max(0)[1]
            # for c in label:
            #     c =c[c.sum(axis=1)!=0]
            #     cout = np.array(c).astype('int')

            #     cv2.drawContours(im,[cout],-1,(0,255,0),1)
            cls =[]
            
            for j in range (len(points)):
                point = points[j]
                adj,feature,offset = self.cal_features1(point,seg,filed)
                point_out,point_offset = self.gcn(feature,adj,offset)
                index = []
            
                point_offset =point_offset.data.cpu().numpy()
                point = point + point_offset
                end = len(point[point.sum(axis=1)!=0,:])
                for k in range(end):
                    if point_out[k] [0]>point_out[k][1]:
                        index.append(k)
                    else:
                            continue
                if index !=[]:
                    cls.append(np.ceil(point[index]).astype('int'))
                iou = []
                for n in range(len(label)):
                    iou.append(self.cal_iou(point,label[n]))
                label_index = np.argmax(iou)
                res = self.points_loss(point,label[label_index])
                pre_point = torch.as_tensor(point[:end].reshape([-1,2]))
                gt_point = torch.as_tensor(label[label_index].reshape([-1,2]))
                set_loss = points_set(pre_point,gt_point)
                cc = torch.LongTensor(res.T).cuda()                
                t_loss= F.nll_loss( point_out, cc)
                s_loss = t_loss + s_loss + set_loss
                # if cls != []:
                #     cv2.drawContours(im,cls,-1,(255,0,255),1)
                # cv2.imwrite('pre_point.png',im)
            if len(points) == 0:
                loss = torch.tensor(1.0) + loss
            else:
                loss = s_loss/len(points)+loss
            out_points.append(cls)
        return loss/(b*300),out_points  

   
    def multi_batch_cal2(self,batch_points,batch_image,seg_out,batch_filed,batch_label):
        loss = torch.tensor(0.0)
        points_set = gcn.ChamferLoss2D()
        b,c,w,h =seg_out.size()
        out_points =[]
        
        #im = np.zeros([w,h,3])

        s_loss = 0.0
        for i in range(b):
            points = batch_points[i]
            filed = batch_filed[i]
            label = batch_label[i]
            seg = seg_out[i].max(0)[1]
            image =batch_image[i]
            # for c in label:
            #     c =c[c.sum(axis=1)!=0]
            #     cout = np.array(c).astype('int')

            #     cv2.drawContours(im,[cout],-1,(0,255,0),1)
            cls =[]
            
            for j in range (len(points)):
                point = points[j]
                adj,feature,offset = self.cal_features2(point,image,seg,filed)
                point_out,point_offset = self.gcn(feature.unsqueeze(0),adj.unsqueeze(0),offset.unsqueeze(0))
                index = []
            
                point_offset =point_offset.data.cpu().numpy()
                point = point + point_offset
                end = len(point[point.sum(axis=1)!=0,:])
                for k in range(end):
                    if point_out[k] [0]>point_out[k][1]:
                        index.append(k)
                    else:
                            continue
                if index !=[]:
                    cls.append(np.ceil(point[index]).astype('int'))
                iou = []
                for n in range(len(label)):
                    iou.append(self.cal_iou(point,label[n]))
                label_index = np.argmax(iou)
                res = self.points_loss(point,label[label_index])
                pre_point = torch.as_tensor(point[:end].reshape([-1,2]))
                gt_point = torch.as_tensor(label[label_index].reshape([-1,2]))
                set_loss = points_set(pre_point,gt_point)
                cc = torch.LongTensor(res.T).cuda()                
                t_loss= F.nll_loss( point_out, cc)
                s_loss = t_loss + s_loss + set_loss
                # if cls != []:
                #     cv2.drawContours(im,cls,-1,(255,0,255),1)
                # cv2.imwrite('pre_point.png',im)
            if len(points) == 0:
                loss = torch.tensor(1.0) + loss
            else:
                loss = s_loss/len(points)+loss
            out_points.append(cls)
        return loss/(b*300),out_points  

    def point_influence(self,batch_points,seg_out,batch_filed):
        b,c,w,h =seg_out.size()
        out_points =[]
        
        #im = np.zeros([w,h,3])

        s_loss = 0.0
        for i in range(b):
            points = batch_points[i]
            filed = batch_filed[i]
            seg = seg_out[i].max(0)[1]

            cls =[]
            
            for j in range (len(points)):
                point = points[j]
                adj,feature,offset = self.cal_features1(point,seg,filed)
                point_out,point_offset = self.gcn(feature,adj,offset)
                index = []
                point_offset =point_offset.data.cpu().numpy()
                point = point + point_offset
                end = len(point[point.sum(axis=1)!=0,:])
                for k in range(end):
                    if point_out[k] [0]>point_out[k][1]:
                        index.append(k)
                    else:
                            continue
                if index !=[]:
                    cls.append(np.ceil(point[index]).astype('int'))
            out_points.append(cls)
        return out_points 

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()              
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
    
    def detect_corners(self,polylines, u, v):
        def compute_direction_score(ij, edges, field_dir):
            values = field_dir[ij[:, 0], ij[:, 1]]
            edge_dot_dir = edges[:, 0] * values.real + edges[:, 1] * values.imag
            abs_edge_dot_dir = np.abs(edge_dot_dir)
            return abs_edge_dot_dir

        def compute_is_corner(points, left_edges, right_edges):
            if points.shape[0] == 0:
                return np.empty(0, dtype=np.bool)

            coords = np.round(points).astype(np.int)
            coords[:, 0] = np.clip(coords[:, 0], 0, u.shape[0] - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, u.shape[1] - 1)
            left_u_score = compute_direction_score(coords, left_edges, u)
            left_v_score = compute_direction_score(coords, left_edges, v)
            right_u_score = compute_direction_score(coords, right_edges, u)
            right_v_score = compute_direction_score(coords, right_edges, v)

            left_is_u_aligned = left_v_score < left_u_score
            right_is_u_aligned = right_v_score < right_u_score

            return np.logical_xor(left_is_u_aligned, right_is_u_aligned)
            
        corner_masks = []
        for polyline in polylines:
            corner_mask = np.zeros(polyline.shape[0], dtype=np.bool)
            if np.max(np.abs(polyline[0] - polyline[-1])) < 1e-6:
                # Closed polyline
                left_edges = np.concatenate([polyline[-2:-1] - polyline[-1:], polyline[:-2] - polyline[1:-1]], axis=0)
                right_edges = polyline[1:] - polyline[:-1]          
                corner_mask[:-1] = compute_is_corner(polyline[:-1, :], left_edges, right_edges)
                # left_edges and right_edges do not include the redundant last vertex, thus we have to do this assignment:
                corner_mask[-1] = corner_mask[0]

            else:
                # Open polyline
                corner_mask[0] = True
                corner_mask[-1] = True
                left_edges = polyline[:-2] - polyline[1:-1]
                right_edges = polyline[2:] - polyline[1:-1]
                corner_mask[1:-1] = compute_is_corner(polyline[1:-1, :], left_edges, right_edges)
            
            corner_masks.append(corner_mask)

        return corner_masks
    def split_polylines_corner(self,polylines, corner_masks):    
        new_polylines = []
        for polyline, corner_mask in zip(polylines, corner_masks):
            splits, = np.where(corner_mask)
            if len(splits) == 0:
                new_polylines.append(polyline)
                continue
            slice_list = [(splits[i], splits[i+1] + 1) for i in range(len(splits) - 1)]
            for s in slice_list:
                new_polylines.append(polyline[s[0]:s[1]])
            # Possibly add a merged polyline if start and end vertices are not corners (or endpoints of open polylines)
            if ~corner_mask[0] and ~corner_mask[-1]:  # In fact any of those conditon should be enough
                new_polyline = np.concatenate([polyline[splits[-1]:], polyline[:splits[0] + 1]], axis=0)
                new_polylines.append(new_polyline)
        return new_polylines

    def corner_detect(self,seg,angle_field,boundarys):
        batch,_,height,width = seg.size()
        out = []
        angle_field = angle_field.data.cpu().numpy()
 
        for i in range(batch):
            seg_res = seg[i]
            seg_res = seg_res.max(0)[1].data.cpu().numpy()
            # boundary = boundarys[i]
            # boundary =boundary.max(0)[1].data.cpu().numpy()
            profile = seg_res
            # contours, hierarchy = cv2.findContours(profile.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
            # res =[]
            # for polygon in contours:             
            #     polygon = self.point_reshape(polygon,30)
            #     res.append(polygon)
            # res =np.array(res).astype('int')
         
            # out.append(res)    
            init_contours= skimage.measure.find_contours(profile, 0.5, fully_connected='low', positive_orientation='high')           
            polygons =[skimage.measure.approximate_polygon(contour, tolerance=min(1, 0.2)) for contour in init_contours]
            conner_mask= self.detect_corners(polygons,angle_field[i][0],angle_field[i][1])
            contours =self.split_polylines_corner(polygons, conner_mask)
            line_string_list = [shapely.geometry.LineString(out_contour[:, ::-1]) for out_contour in contours]
            line_string_list = [line_string.simplify(1.0, preserve_topology=True) for line_string in line_string_list]


                # Add image boundary line_strings for border polygons
            line_string_list.append(
                    shapely.geometry.LinearRing([
                        (0, 0),
                        (0, height-1 ),
                        (width-1, height-1),
                        (width-1, 0),
                    ]))


            multi_line_string = shapely.ops.unary_union(line_string_list)

            polygons = shapely.ops.polygonize(multi_line_string)
            filtered_polygons = []
   
            for polygon in polygons:
                prob = self.compute_geom_prob(polygon, profile)
                # print("acm:", np_indicator.min(), np_indicator.mean(), np_indicator.max(), prob)
                if 0.5 < prob:
                    filtered_polygons.append(polygon)

            res =[]
            for polygon in filtered_polygons:
                polygon = list(polygon.exterior.coords)
                polygon = self.point_reshape(polygon,30)
                res.append(polygon)
            res =np.array(res).astype('int')
           # polygons=[list(polygon.exterior.coords) for polygon  in polygons ]
            out.append(res)
          
    
        return out
    def point_reshape(self,points,normallength):       
        dp_threshold = 5
        points = np.array(points).reshape([-1,2])
        
        out = points
        while(normallength!=len(points)):            
            if (len(points)< normallength):
                add = np.zeros([normallength-len(points),2])                   
                out = np.concatenate([out,add],axis=0)
                break
            elif normallength==len(points):
                out = points.reshape([-1,2])
                break
            else:
                dp_points = cv2.approxPolyDP(np.array(points).astype('int') , dp_threshold, True)
                points = dp_points
                dp_threshold = dp_threshold +2
                out = points.reshape([-1,2])
        return out 
    def compute_geom_prob(self,geom, prob_map, output_debug=False):
      
            assert len(np.shape(prob_map)) == 2, "prob_map should have size (H, W), not {}".format(prob_map.shape)
            
            # --- Cut with geom bounds:
            minx, miny, maxx, maxy = geom.bounds
            minx = int(minx)
            miny = int(miny)
            maxx = int(maxx) + 1
            maxy = int(maxy) + 1
            geom = shapely.affinity.translate(geom, xoff=-minx, yoff=-miny)
            prob_map = prob_map[miny:maxy, minx:maxx]

            # --- Rasterize TODO: better rasterization (or sampling) of polygon ?
            raster = np.zeros(np.shape(prob_map), dtype=np.uint8)
            exterior_array = np.round(np.array(geom.exterior.coords)).astype(np.int32)
            interior_array_list = [np.round(np.array(interior.coords)).astype(np.int32) for interior in geom.interiors]
            cv2.fillPoly(raster, [exterior_array], color=1)
            cv2.fillPoly(raster, interior_array_list, color=0)

            raster_sum = np.sum(raster)
            if 0 < raster_sum:
                polygon_prob = np.sum(raster * prob_map) / raster_sum
            else:
                polygon_prob = 0
             
            return polygon_prob
  
def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg.MODEL.PRETRAINED)

    return model

if __name__ =="__main__":
    f  =open("config/HRnet.yaml",'r',encoding='utf-8')
    cont = f.read()
    config = yaml.load(cont)
    model=HighResolutionNet(config).cuda()
    im =cv2.imread("tst.png")
    im = torch.tensor(im).cuda()
    im= im.permute(2,0,1)
    input = im.unsqueeze(0)
    model.eval()
    input = torch.rand(1, 3,300, 300).cuda()
    point_out,c ,b= model(input/255)
    print(c.size())
    

