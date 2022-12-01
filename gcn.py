from matplotlib.pyplot import axis
import numpy as np
import scipy.sparse as sp
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import cv2
import scipy.spatial as spt
import math

from zmq import device
from datasets.loss import *
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    
    return mx

def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
    np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features),requires_grad=True)
        self.use_bias = bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features),requires_grad=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, input_features, adj):
        support = torch.mm(input_features, self.weight)
        output = torch.spmm(adj, support)
        if self.use_bias:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, input_dim=[9,4]):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim[0], 16)
        self.gcn2 = GraphConvolution(16, 2)
        self.gon1 = GraphConvolution(input_dim[1], 16)
        self.gon2 = GraphConvolution(16, 2)
        self.out =  nn.LeakyReLU(inplace=True)
        pass
    
    def forward(self, X, adj,offset):
        Y = F.relu(self.gcn1(X, adj))
        O = F.relu(self.gon1(offset, adj))

        Y = self.gcn2(Y, adj)
        O = self.gon2(O, adj)

        return F.log_softmax(Y, dim=1),self.out(O)
    

def cal_feature(points,seg,filed):
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
                feature = [c_x, c_y]            
                feature1 = torch.stack([pad_seg[x-step, y-step], pad_seg[x, y-step],pad_seg[x, y+step],
                pad_seg[x-step, y],pad_seg[x, y], pad_seg[x+step, y],pad_seg[x-step, y+step], 
                pad_seg[x, y+step], pad_seg[x+step, y+step]] ,0) 
                feature2=  torch.stack([pad_filed_all[x-step, y-step], pad_filed_all[x, y-step],pad_filed_all[x, y+step],
                pad_filed_all[x-step, y],pad_filed_all[x, y], pad_filed_all[x+step, y],pad_filed_all[x-step, y+step], 
                pad_filed_all[x, y+step], pad_filed_all[x+step, y+step]] ,0)  
                X.append(feature1)
                O.append(feature2)
            else:
                feature = torch.FloatTensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]).cuda()
                X.append(feature)
                O.append(feature)
        X = torch.stack(X,0)
        A = torch.from_numpy(np.asarray(A)).float().cuda()
        O = torch.stack(O,0)
        return A, X,O
# def cal_feature(points,image,seg,filed):
#         step = 1
#         w,h = seg.size() 
#         pad_seg = torch.zeros([w+2*step,h+2*step])
#         pad_seg[step:step+w,step:step+w] =seg

#         pad_filed = torch.zeros([2,w+2*step,h+2*step])
#         pad_filed[:,step:step+w,step:step+w]=filed
#         pad_filed = pad_filed

#         pad_image = torch.zeros([3,w+2*step,h+2*step])
#         pad_image[:,step:step+w,step:step+w] =image
#         G = nx.Graph(name="G")
#         # 创建节点
#         end =len(points[points.sum(axis=1)!=0])
#         edges = []
#         for i in range(len(points)):
#             G.add_node(i, name=i)
#         for i in range(end):
#             if not i ==  end-1:
#                 edge = (i, i+1) 
#                 edges.append(edge)
#             else:
#                 edge = (i, 0)
#                 edges.append(edge)               
#         # for i in range(len(points)):
#         #     if points[i][1]+points[i][0]!=0:
#         #         G.add_node(i, name=i)
#         #         if not i ==  end-1:
#         #             edge = (i, i+1)
#         #             edges.append(edge)
#         #         else:
#         #             edge = (i, 0)
#         #             edges.append(edge)
#         #     else:
#         #         G.add_node(i, name=i)
#         # 创建边并添加边到图里
#         G.add_edges_from(edges)
#         # 从图G获得邻接矩阵（A）和节点特征矩阵（X）
#         # nx.draw(G,with_labels=True,font_weight='bold')
#         # plt.show()
#         A = np.array(nx.attr_matrix(G, node_attr='name')[0])
#         X = torch.zeros([len(points),36]).cuda()
#         O = torch.zeros([len(points),18]).cuda()
        
#         for j in range(len(points)):
#             c_x, c_y = points[j]
#             x,y = math.floor(c_x)+step,math.floor(c_y)+step
#             # x = int(x.data.cpu().numpy())
#             # y = int(y.data.cpu().numpy())
#             if not j ==  end-1:
#                 feature = [c_x, c_y]            
#                 feature1 = [pad_seg[x-step, y-step], pad_seg[x, y-step],pad_seg[x, y+step],
#                 pad_seg[x-step, y],pad_seg[x, y], pad_seg[x+step, y],pad_seg[x-step, y+step], 
#                 pad_seg[x, y+step], pad_seg[x+step, y+step],
#                 pad_image[0,x-step, y-step], pad_image[0,x, y-step],pad_image[0,x, y+step],
#                 pad_image[0,x-step, y],pad_image[0,x, y], pad_image[0,x+step, y],
#                 pad_image[0,x-step, y+step],pad_image[0,x, y+step], pad_image[0,x+step, y+step],
#                 pad_image[1,x-step, y-step], pad_image[1,x, y-step],pad_image[1,x, y+step],
#                 pad_image[1,x-step, y],pad_image[1,x, y], pad_image[1,x+step, y],
#                 pad_image[1,x-step, y+step], pad_image[1,x, y+step], pad_image[1,x+step, y+step],
#                 pad_image[2,x-step, y-step], pad_image[2,x, y-step],pad_image[2,x, y+step],
#                 pad_image[2,x-step, y],pad_image[2,x, y], pad_image[2,x+step, y],
#                 pad_image[2,x-step, y+step], pad_image[2,x, y+step], pad_image[2,x+step, y+step],]  
                
#                 feature2=  [pad_filed[0][x-step, y-step], pad_filed[0][x, y-step],pad_filed[0][x, y+step],
#                 pad_filed[0][x-step, y],pad_filed[0][x, y], pad_filed[0][x+step, y],pad_filed[0][x-step, y+step], 
#                 pad_filed[0][x, y+step], pad_filed[0][x+step, y+step],pad_filed[1][x-step, y-step], pad_filed[1][x, y-step],pad_filed[1][x, y+step],
#                 pad_filed[1][x-step, y],pad_filed[1][x, y], pad_filed[1][x+step, y],pad_filed[1][x-step, y+step], 
#                 pad_filed[1][x, y+step], pad_filed[1][x+step, y+step]]  
#                 X[j,:] = torch.stack(feature1)
#                 O[j,:] = torch.stack(feature2)
#             else:
#                 X[j,:] = torch.zeros([36])
#                 O[j,:] = torch.zeros([18])
 

#         A = torch.from_numpy(np.asarray(A)).float().cuda()

#         return A, X,O

def points_loss(point,label):
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

def cal_iou(point,label):
  
    point = point[point.sum(axis=1)!=0,:]
    px1,py1 = np.min(point[:,0]),np.min(point[:,1])
    px2,py2 = np.max(point[:,0]),np.max(point[:,1])
   
    gx1,gy1 = np.min(label[:,0]),np.min(label[:,1])
    gx2,gy2 = np.max(label[:,0]),np.max(label[:,1])

 
    parea = (px2 - px1) * (py2 - py1) # 计算P的面积
    garea = (gx2 - gx1) * (gy2 - gy1) # 计算G的面积

    
    # 求相交矩形的左上和右下顶点坐标(x1, y1, x2, y2)
    x1 = max(px1, gx1) # 得到左上顶点的横坐标
    y1 = min(py1, gy1) # 得到左上顶点的纵坐标
    x2 = min(px2, gx2) # 得到右下顶点的横坐标
    y2 = max(py2, gy2) # 得到右下顶点的纵坐标
    
    # 利用max()方法处理两个矩形没有交集的情况,当没有交集时,w或者h取0,比较巧妙的处理方法
    # w = max(0, (x2 - x1)) # 相交矩形的长，这里用w来表示
    # h = max(0, (y1 - y2)) # 相交矩形的宽，这里用h来表示
    # print("相交矩形的长是：{}，宽是：{}".format(w, h))
    # 这里也可以考虑引入if判断
    w = x2 - x1
    h = y2 - y1
    if w <=0 or h <= 0:
        return 0       
    area = w * h # G∩P的面积   
    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)  
    return IoU

def multi_batch_cal(batch_points,seg_out,batch_filed,model,batch_label):
    loss = torch.tensor(0.0)
    points_set = ChamferLoss2D()
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
            adj,feature,offset = cal_feature(point,seg,filed)
            point_out,point_offset = model(feature,adj,offset)
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
                iou.append(cal_iou(point,label[n]))
            label_index = np.argmax(iou)
            res = points_loss(point,label[label_index])
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



if __name__ == "__main__":
    img = np.zeros([500, 500]).astype(np.uint8)
    seg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    label= np.array([[10, 10], [10, 200], [200, 200], [200, 10]])
    cv2.drawContours(seg, [label], -1, (255, 255, 255), -1)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('s.png', seg)
    label = [[10, 10], [10, 200], [200, 200], [200, 10]]
    points = [[10, 10], [10,100],[10, 200], [200, 200], [200, 10]] 
    adj,features = cal_feature(points,gray)   
    model = GCN(features.shape[1])
    aa = torch.from_numpy(np.asarray(features)).float()
    bb = torch.from_numpy(np.asarray(adj)).float()
    output = model(aa,bb)
    output = output.max(1)[1]
    res = [points[i]  for i in range(len(points)) if output[i] == 1]
    cc = points_loss([points],[label])   
    cc= torch.LongTensor(cc.T)
    print( F.nll_loss(output, cc))

