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
from dataset.loss import *
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
    def __init__(self, input_dim=1433):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 16)
        self.gcn2 = GraphConvolution(16, 2)
        pass
    
    def forward(self, X, adj):
        X = F.relu(self.gcn1(X, adj))
        X = self.gcn2(X, adj)
        
        return F.log_softmax(X, dim=1)
    

def cal_feature(points,seg):
        step = 1
        w,h = seg.size() 
        pad_seg = torch.zeros([w+2*step,h+2*step])
        pad_seg[step:step+w,step:step+w] =seg
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
        for p in points:
            c_x, c_y = p
            x,y = math.floor(c_x)+step,math.floor(c_y)+step
            # x = int(x.data.cpu().numpy())
            # y = int(y.data.cpu().numpy())
            if x+y !=0:
                feature = [c_x, c_y]            
                feature = feature + [ pad_seg[x-step, y-step], pad_seg[x, y-step],pad_seg[x, y+step],
                pad_seg[x-step, y],pad_seg[x, y], pad_seg[x+step, y],pad_seg[x-step, y+step], 
                pad_seg[x, y+step], pad_seg[x+step, y+step]]       
                X.append(feature)
            else:
                feature = [c_x, c_y]+[0 for i in range(9)]
                X.append(feature)
        X = torch.tensor(np.array(X)).float()
        A = torch.from_numpy(np.asarray(A)).float()
        return A, X

def points_loss(point,label):
    n_points = point
    l_points = label.data.cpu().numpy()
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
    label =label.data.cpu().numpy()
    point = point[point.sum(axis=1)!=0,:]
    px1,py1 = np.min(point[:,0]),np.min(point[:,1])
    px2,py2 = np.max(point[:,0]),np.max(point[:,1])
   
    gx1,gy1 = np.min(label[:,0]),np.min(label[:,1])
    gx2,gy2 = np.max(label[:,0]),np.max(label[:,1])

 
    parea = (px2 - px1) * (py1 - py2) # 计算P的面积
    garea = (gx2 - gx1) * (gy1 - gy2) # 计算G的面积

    
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
    h = y1 - y2
    if w <=0 or h <= 0:
        return 0       
    area = w * h # G∩P的面积   
    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)  
    return IoU

def multi_batch_cal(points,seg_out,model,label):
    loss = torch.tensor(0.0)
    points_set = ChamferLoss2D()
    b,c,w,h =seg_out.size()
    out_points =[]
    for i in range(len(points)):
        
        seg =seg_out[i].max(0)[1]
        s_loss = torch.tensor(0.0)
        im_point = []
        im = np.zeros([w,h,3])
        for p  in points[i]:
            im_point.append(p[p.sum(axis=1)!=0,:])
        cv2.drawContours(im,im_point,-1,(255,255,255),1)
        cls =[]
        for j in range (len(points[i])):
            point = points[i][j]
            adj,feature = cal_feature(point,seg)
            point_out = model(feature,adj)
            index = []
            end = len(point[point.sum(axis=1)!=0,:])
            for k in range(end):
                if point_out[k] [0]>point_out[k][1]:
                    index.append(k)
                else:
                    continue
            if index !=[]:
                cls.append(point[index])
            iou = []
            for n in range(len(label)):
                iou.append(cal_iou(point,label[n]))
            label_index = np.argmax(iou)
            res = points_loss(point,label[i][label_index])
            pre_point = torch.tensor(point[:end].reshape([-1,2]))
            gt_point = label[i][label_index].reshape([-1,2])
            set_loss = points_set(pre_point,gt_point)
            cc = torch.LongTensor(res.T)                
            t_loss= F.nll_loss( point_out, cc)
            s_loss = t_loss + s_loss + set_loss
        cv2.drawContours(im,cls,-1,(255,0,255),1)
        cv2.imwrite('pre_point.png',im)
        if len(points[i]) == 0:
            loss = torch.tensor(10.0) + loss
        else:
            loss = s_loss/len(points[i])+loss
        out_points.append(cls)
    return loss/(len(points)),out_points               



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

