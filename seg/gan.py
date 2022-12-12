import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import cv2
import math
import scipy.spatial as spt
from datasets.loss import *
 
def get_weights(size, gain=1.414):
    weights = nn.Parameter(torch.zeros(size=size))
    nn.init.xavier_uniform_(weights, gain=gain)
    return weights
 
class GraphAttentionLayer(nn.Module):
    '''
    Simple GAT layer 图注意力层 (inductive graph)
    '''
    def __init__(self, in_features, out_features,  alpha, concat = True, head_id = 0):
        ''' One head GAT '''
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  #节点表示向量的输入特征维度
        self.out_features = out_features    #节点表示向量的输出特征维度
       
        self.alpha = alpha  #leakyrelu激活的参数
        self.concat = concat    #如果为true，再进行elu激活
        self.head_id = head_id  #表示多头注意力的编号
 
        self.W_type = nn.ParameterList()
        self.a_type = nn.ParameterList()
        self.n_type = 1 #表示边的种类
        for i in range(self.n_type):
            self.W_type.append(get_weights((in_features, out_features)))
            self.a_type.append(get_weights((out_features * 2, 1)))
 
        #定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size = (in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain = 1.414)  #xavier初始化
        self.a = nn.Parameter(torch.zeros(size = (2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain = 1.414)  #xavier初始化
 

        #定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
 
    def forward(self, node_input, adj, node_mask = None):
        '''
        node_input: [batch_size, node_num, feature_size] feature_size 表示节点的输入特征向量维度
        adj: [batch_size, node_num, node_num] 图的邻接矩阵
        node_mask:  [batch_size, node_mask]
        '''
 
        zero_vec = torch.zeros_like(adj)
        scores = torch.zeros_like(adj)
 
        for i in range(self.n_type):
            h = torch.matmul(node_input, self.W_type[i])

            N, E, d = h.shape   # N == batch_size, E == node_num, d == feature_size
            a_input = torch.cat([h.repeat(1, 1, E).view(N, E * E, -1), h.repeat(1, E, 1)], dim = -1)
            a_input = a_input.view(-1, E, E, d*2)     #([batch_size, E, E, out_features])
 
            score = self.leakyrelu(torch.matmul(a_input, self.a_type[i]).squeeze(-1))   #([batch_size, E, E, 1]) => ([batch_size, E, E])
            #图注意力相关系数（未归一化）
 
            zero_vec = zero_vec.to(score.dtype)
            scores = scores.to(score.dtype)
            scores += torch.where(adj == i+1, score, zero_vec.to(score.dtype))
 
        zero_vec = -1*30 * torch.ones_like(scores)  #将没有连接的边置为负无穷
        attention = torch.where(adj > 0, scores, zero_vec.to(scores.dtype))    #([batch_size, E, E])
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，则该位置的注意力系数保留；否则需要mask并置为非常小的值，softmax的时候最小值不会被考虑
 
        if node_mask is not None:
            node_mask = node_mask.unsqueeze(-1)
            h = h * node_mask   #对结点进行mask
 
        attention = F.softmax(attention, dim = 2)   #[batch_size, E, E], softmax之后形状保持不变，得到归一化的注意力权重
        h = attention.unsqueeze(3) * h.unsqueeze(2) #[batch_size, E, E, d]
        h_prime = torch.sum(h, dim = 1)             #[batch_size, E, d]
 
        # h_prime = torch.matmul(attention, h)    #[batch_size, E, E] * [batch_size, E, d] => [batch_size, N, d]
 
        #得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
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
    def __init__(self, in_dim, hid_dim, alpha, n_heads, concat = True):
        '''
        Dense version of GAT
        in_dim输入表示的特征维度、hid_dim输出表示的特征维度
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似于self-attention从不同的子空间进行抽取特征
        '''
        super(GCN, self).__init__()
    
        
        self.alpha = alpha
        self.concat = concat
 
        self.attn_funcs1 = nn.ModuleList()
        for i in range(n_heads):
            self.attn_funcs1.append(
                #定义multi-head的图注意力层
                GraphAttentionLayer(in_features = in_dim[0], out_features = hid_dim[0] // n_heads,
                                     alpha = alpha, concat = concat, head_id = i)
            )
        self.attn_funcs2 = nn.ModuleList()
        for i in range(n_heads):
            self.attn_funcs2.append(
                #定义multi-head的图注意力层
                GraphAttentionLayer(in_features = in_dim[1], out_features = hid_dim [1]// n_heads,
                                     alpha = alpha, concat = concat, head_id = i)
            )
        if in_dim == [9,9]:
            #GCN_NUM:[9,9]
            self.gcn1 = GraphConvolution(8, 16)
            self.gcn2 = GraphConvolution(16, 2)
            self.gon1 = GraphConvolution(8, 16)
            self.gon2 = GraphConvolution(16, 2)   
            self.out =  nn.LeakyReLU(inplace=True)
        else:
            #GCN_NUM:[36,18]
            self.gcn1 = GraphConvolution(18, 16)
            self.gcn2 = GraphConvolution(16, 2)
            self.gon1 = GraphConvolution(18, 16)
            self.gon2 = GraphConvolution(16, 2)   
            self.out =  nn.LeakyReLU(inplace=True)
        


    def forward(self, node_input,adj,uv_filed,  node_mask = None):
        '''
        node_input: [batch_size, node_num, feature_size]    输入图中结点的特征
        adj:    [batch_size, node_num, node_num]    图邻接矩阵
        node_mask:  [batch_size, node_num]  表示输入节点是否被mask
        '''
        hidden_list1 = []
        hidden_list2 = []
        for attn in self.attn_funcs1:
            h = attn(node_input, adj, node_mask = node_mask)
            hidden_list1.append(h)
        for attn in self.attn_funcs2:
            h = attn(uv_filed, adj, node_mask = node_mask)
            hidden_list2.append(h)
        
        h = torch.cat(hidden_list1, dim = -1)    
      
        o= torch.cat(hidden_list2, dim = -1)
        
        Y = F.relu(self.gcn1(h[0], adj[0]))
        O = F.relu(self.gon1(o[0], adj[0]))

        Y = self.gcn2(Y, adj[0])
        O = self.gon2(O, adj[0])
        return F.log_softmax(Y, dim=1),self.out(O)
 
 
def cal_feature(points,seg,filed):
        step = 1
        w,h = seg.size() 
        pad = nn.ZeroPad2d(padding=(2*step, 2*step, 2*step, 2*step))
        pad_seg = pad(seg)
            
        pad_filed = pad(filed)
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
    l_points = label.data.cpu().numpy()
  

    kt = spt.KDTree(l_points, leafsize=10)
    dict_l={}
    if 2>= len(point):
        return 
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
   
    gx1,gy1 =np.min(label[:,0]),np.min(label[:,1])
    gx2,gy2 =np.max(label[:,0]),np.max(label[:,1])

 
    s1 = (px2 - px1) * (py2 - py1) # C的面积
    s2 = (gx2 - gx1) * (gy2 - gy1) # G的面积
    
    # 计算相交矩形
    xmin = max(px1, gx1)
    ymin = max(py1, gy1)
    xmax = min(px2, gx2)
    ymax = min(py2, gy2)
    
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou

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
            point_out,point_offset = model(feature,offset,adj)
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
 
