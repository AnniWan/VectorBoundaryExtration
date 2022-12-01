
from this import d
from tokenize import _all_string_prefixes
import numpy as np
import shapely
import torch
import cv2
import scipy.spatial as spt
import torch
import torch.nn as nn
import skimage.morphology
import shapely.geometry
import shapely.ops
import shapely.prepared
import datasets.angle as angle

import torch.nn.functional as F
#from mmdet.ops.chamfer_2d import Chamfer2D

def instect(array1,array2):
    assert np.shape(array1) == np.shape(array2)
    out=np.zeros(np.shape(array1))


    for i in range(len(array1)):
        for j in range(len(array1[i])):
           if array1[i][j] != 0  and array2[i][j] !=0:
                out[i][j] = ((array1[i][j]-array2[i][j])**2)
              
           else:
                out[i][j] = 0
    return out  

def points_loss(input, label):
    batch = len(input)
    out = []
    for i in range(batch):
        n_points = input[i]
        l_points = label[i]
        kt = spt.KDTree(l_points, leafsize=10)
        dict_l = {}
        for j in range(len(n_points)):
            # 去掉补零数据的影响
            if 0.0005 >= n_points[j][0] and 0.0005 >= n_points[j][1]:
                continue
            d, x = kt.query(n_points[j])

            if str(l_points[x]) not in dict_l.keys():
                dict_l[str(l_points[x])] = [[j, d]]
            else:
                dict_l[str(l_points[x])].append([j, d])
        res = np.zeros(len(n_points))
        for key, values in dict_l.items():
            values = sorted(values, key=(lambda x: x[1]))
            res[values[0][0]] = 1
        out.append(res)
    return out


def ce_loss(seg, gt_seg):
    loss = torch.nn.CrossEntropyLoss()
    return loss(seg, gt_seg)


def con_loss(field,contours):
    b,c,h,w = contours.size()
    loss =  torch.tensor(0.0).cuda()
    for i in range(b):
        points = field[i].data.cpu().numpy()
        contour = contours[i][0].data.cpu().numpy()
        cv2.imwrite("field.png",points*255)
        loss_s = instect(points,contour)
        loss = np.sum(loss_s)/(h*w)+loss
    
    loss  = loss/b
    return torch.as_tensor(loss).cuda()
    # b,c,h,w = contours.size()

    # loss = 0.0
    # im = np.zeros([h,w,3])
    
    # for i in range(len(gt_points)):
    #     points = gt_points[i].data.cpu().numpy()
    #     cv2.drawContours(im,points,-1,(255,255,255),1)
    # gray = cv2.cvtColor(im.astype('uint8'),cv2.COLOR_BGR2GRAY) 
    # cv2.imwrite("gt_point.png",gray)
    # contour = contours
    # gt_bound = torch.LongTensor(gray/255).unsqueeze(0).cuda()
    # loss = ce_loss(contour.unsqueeze(0),gt_bound)           

    # return torch.as_tensor(loss/(h*w)).cuda()

def con_loss2(bound,target):
    
    b,c,h,w = bound.size()
    t_loss =  torch.tensor(0.0).cuda()
    for i in range(b):
        im = np.zeros([h,w,3])
        boundary = bound[i].max(0)[1] 
        gt_points = target[i] 
        cv2.drawContours(im,gt_points,-1,(255,255,255),2)
        gray = cv2.cvtColor(im.astype('uint8'),cv2.COLOR_BGR2GRAY) 
       # cv2.imwrite("gt_point.png",gray)

        gt_bound = torch.as_tensor(gray/255).cuda()
        loss = F.mse_loss(boundary,gt_bound)        
        # boundary = boundary.max(0)[1].data.cpu().numpy()*255
        # cv2.imwrite("detect_bounary.png",boundary)
        t_loss = t_loss +loss 
    loss = t_loss/b
    return loss.long()
    
    # b,c,h,w = bound.size()
    # t_loss =  torch.tensor(0.0).cuda()
    # for i in range(b):
    #     boundary = bound[i].max(0)[1]  
    #     seg = target[i].data.cpu().numpy()*255
    #     seg = seg.astype('uint8')
    #     out = cv2.Canny(seg, 50, 150) 
        
    #     # cv2.imwrite("gt_bounary.png",out)
    #     gt_bound = torch.as_tensor(out/255).cuda()
    #     loss = F.mse_loss(boundary,gt_bound)        
    #     # boundary = boundary.max(0)[1].data.cpu().numpy()*255
    #     # cv2.imwrite("detect_bounary.png",boundary)
    #     t_loss = t_loss +loss 
    # loss = t_loss/b
    # return loss.long()

def field_loss(field,field_out):
    b,c,h,w = field_out.size()
    loss =  torch.tensor(0.0).cuda()
    for i in range(b):
        gradx_loss = F.mse_loss(field_out[i][0],field[0][i].type(torch.float32))
        grady_loss = F.mse_loss(field_out[i][1],field[1][i].type(torch.float32))
        # cv2.imwrite("field_x.png",field_out[i][0]*255)
        # cv2.imwrite("field_y.png",field_out[i][1]*255)       
        loss = (gradx_loss+grady_loss)+loss
    loss  = loss/b
    return loss

def BD_loss(net_output, bound):
    b,c,h,w = net_output.size()
    loss =  torch.tensor(0.0).cuda()
    for i in range(b):
        seg = net_output[i].max(0)[1]
       
        seg = seg.data.cpu().numpy()*255
        seg = seg.astype('uint8')
     

        polygons, heridency = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [np.reshape(p,[-1,2]) for p in polygons if len(p) >2 ]
     
        #0410
        out =  angle.init_angle_field(polygons,[seg.shape[0],seg.shape[1]], line_width=1)        
        #0415
        grad_x = cv2.Sobel(out, cv2.CV_32F, 1, 0)  # 对x求一阶导
        grad_y = cv2.Sobel(out, cv2.CV_32F, 0, 1)
        gradx = cv2.convertScaleAbs(grad_x)  
        grady = cv2.convertScaleAbs(grad_y) 
        # cv2.imwrite("seg_x.png",gradx)
        # cv2.imwrite("seg_y.png",grady)  
        gradx =  torch.as_tensor(gradx).cuda()
        grady =  torch.as_tensor(grady).cuda()
        gradx_loss = F.mse_loss(gradx/255,bound[i][0])
        grady_loss = F.mse_loss(grady/255,bound[i][1])
 
      #  cv2.imwrite("seg_bounary.png",out)
        loss = (gradx_loss+grady_loss)+loss
    loss  = loss/b
    return loss


def BD_loss2(net_output, bound):
    b,c,h,w = net_output.size()
    t_loss = torch.tensor(0.0).cuda()
    for i in range(b):
        seg = net_output[i].max(0)[1]
        boundary = bound[i].max(0)[1]
        
        seg = seg.data.cpu().numpy()*255
        seg = seg.astype('uint8')
        out = cv2.Canny(seg, 50, 150) 
        
      #  cv2.imwrite("seg_bounary.png",out)
        gt_bound = torch.as_tensor(out/255).cuda()
        loss = F.mse_loss(boundary,gt_bound)        
       # boundary = boundary.data.cpu().numpy()*255
      #  cv2.imwrite("detect_bounary.png",boundary)
        t_loss = t_loss +loss 
    loss = t_loss/b
    return loss.long()

class ChamferDistancePytorch(nn.Module):
    r"""
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, reduction='none'):
        super(ChamferDistancePytorch, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        if x.shape[0] == 0:
            return x.sum()
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function

        # compute chamfer loss
        min_x2y, _ = C.min(-1)
        d1 = min_x2y.mean(-1)
        min_y2x, _ = C.min(-2)
        d2 = min_y2x.mean(-1)
        cost = (d1 + d2) / 2.0

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2).float()
        y_lin = y.unsqueeze(-3).float()
        C = torch.norm(x_col - y_lin, 2, -1)
        return C


class ChamferLoss2D(nn.Module):
    def __init__(self, use_cuda=False, loss_weight=1.0, eps=1e-12):
        super(ChamferLoss2D, self).__init__()
        self.use_cuda = use_cuda
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, point_set_1, point_set_2):
        """
        Computation of optimal transport distance via sinkhorn algorithm.
        - Input:
            - point_set_1:	torch.Tensor	[..., num_points_1, point_dim] e.g. [bs, h, w, 1000, 2]; [bs, 1000, 2]; [1000, 2]
            - point_set_2:	torch.Tensor	[..., num_points_2, point_dim]
                    (the dimensions of point_set_2 except the last two should be the same as point_set_1)
        - Output:
            - distance:	torch.Tensor	[...] e.g. [bs, h, w]; [bs]; []
        """
        chamfer =  ChamferDistancePytorch()

        assert point_set_1.dim() == point_set_2.dim()
        assert point_set_1.shape[-1] == point_set_2.shape[-1]
        if point_set_1.dim() <= 3:
            if self.use_cuda:
                dist1, dist2, _, _ = chamfer(point_set_1, point_set_2)
                dist1 = torch.sqrt(torch.clamp(dist1, self.eps))
                dist2 = torch.sqrt(torch.clamp(dist2, self.eps))
                dist = (dist1.mean(-1) + dist2.mean(-1)) / 2.0
            else:
                dist = chamfer(point_set_1, point_set_2)
        else:
            point_dim = point_set_1.shape[-1]
            num_points_1, num_points_2 = point_set_1.shape[-2], point_set_2.shape[-2]
            point_set_1t = point_set_1.reshape((-1, num_points_1, point_dim))
            point_set_2t = point_set_2.reshape((-1, num_points_2, point_dim))
            if self.use_cuda:
                dist1, dist2, _, _ = chamfer(point_set_1, point_set_2)
                dist1 = torch.sqrt(torch.clamp(dist1, self.eps))
                dist2 = torch.sqrt(torch.clamp(dist2, self.eps))
                dist_t = (dist1.mean(-1) + dist2.mean(-1)) / 2.0
            else:
                dist_t = chamfer(point_set_1t, point_set_2t)
            dist_dim = point_set_1.shape[:-2]
            dist = dist_t.reshape(dist_dim)
        return dist * self.loss_weight