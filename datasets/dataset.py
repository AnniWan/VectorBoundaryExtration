import os
import os.path
import cv2
from matplotlib.pyplot import axis
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import shapely.geometry
import shapely.affinity
from PIL import Image, ImageDraw, ImageFilter

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

def point_resize(points,y_scale,x_scale):

        points[:,1]=(y_scale*points[:,1])
        points[:,0]=(x_scale*points[:,0])
        return points
def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']

    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 3:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = data_root+line_split[0]
            label_name = data_root+line_split[1]
            point_name = data_root+line_split[2]    
        else:
            if len(line_split) != 3:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
           # image_name = os.path.join(data_root, line_split[0])
           # label_name = os.path.join(data_root, line_split[1])
           # point_name = os.path.join(data_root,line_split[2])
            image_name = data_root+line_split[0]
            label_name = data_root+line_split[1]
            point_name = data_root+line_split[2]           
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        
        item = (image_name, label_name,point_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class PolyData(Dataset):
    def __init__(self, split='train', point_length=30,data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.point_length =point_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path,point_path = self.data_list[index]
        o_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        name = os.path.basename(image_path)
        image = cv2.cvtColor(o_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        data = json.load(open(point_path ))
        gray_im = np.zeros(image.shape)

        shapes= data["shapes"]
        points =[]
        import random
        width= label.shape[0]
        height=image.shape[1]
        if self.split =='train':
            reminder = random.randint(0, 5)
            if len(shapes)>0:
                if reminder == 1:
                    image = image[:, ::-1, :]
                    label= np.fliplr(label)
                    for shape in shapes:
                        point =  np.array(shape["points"] )                     
                        point[:, 0] = width - point[:, 0]
                        contours = point.astype('int') 
                        cv2.drawContours(gray_im, [contours], -1,(255,255,255),-1)
                        poly = self.point_reshape(contours,o_image,[300,300],self.point_length)           
                        points.append(poly) 
                elif reminder == 2:
                    image = image[::-1, :, :]
                    label= np.flipud(label)
                    for shape in shapes:
                        point = np.array(shape["points"] )                   
                        point[:, 1] = height - point[:, 1]
                        contours = point.astype('int') 
                        cv2.drawContours(gray_im, [contours], -1,(255,255,255),-1)
                        poly = self.point_reshape(contours,o_image,[300,300],self.point_length)           
                        points.append(poly)                     
                elif reminder == 3: # horizontal and vertical flip
                    image = image[::-1, ::-1, :]
                    label = np.fliplr(label)
                    label = np.flipud(label)
                    for shape in shapes:
                        point =  np.array(shape["points"] ) 
                        point[:, 0] = width -point[:, 0]
                        point[:, 1] = height - point[:, 1]
                        contours = point.astype('int') 
                        cv2.drawContours(gray_im, [contours], -1,(255,255,255),-1)
                        poly = self.point_reshape(contours,o_image,[300,300],self.point_length)           
                        points.append(poly)  
                elif reminder == 4: # rotate 90 degree
                    rot_matrix = cv2.getRotationMatrix2D((int(width/2), (height/2)), 90, 1)
                    image = cv2.warpAffine(image, rot_matrix, (width, height))
                    label = cv2.warpAffine(label, rot_matrix, (width, height))
                    for shape in shapes:
                        point =  np.array(shape["points"] )   
                        point = np.asarray([affine_transform(p, rot_matrix) for p in point], dtype=np.float32) 
                        contours = point.astype('int') 
                        cv2.drawContours(gray_im, [contours], -1,(255,255,255),-1)
                        poly = self.point_reshape(contours,o_image,[300,300],self.point_length)           
                        points.append(poly) 
                elif reminder == 5: # rotate 270 degree
                    rot_matrix = cv2.getRotationMatrix2D((int(width / 2), (height / 2)), 270, 1)
                    image = cv2.warpAffine(image, rot_matrix, (width, height))
                    label = cv2.warpAffine(label, rot_matrix, (width, height))
                    for shape in shapes:
                        point =  np.array(shape["points"] )    
                        point = np.asarray([affine_transform(p, rot_matrix) for p in point], dtype=np.float32)         
                        contours = point.astype('int') 
                        cv2.drawContours(gray_im, [contours], -1,(255,255,255),-1)
                        poly = self.point_reshape(contours,o_image,[300,300],self.point_length)           
                        points.append(poly)                
                else:  
                    for shape in shapes:
                        point =  np.array(shape["points"] )       
                        contours = point.astype('int') 
                        cv2.drawContours(gray_im, [contours], -1,(255,255,255),-1)
                        poly = self.point_reshape(contours,o_image,[300,300],self.point_length)           
                        points.append(poly)  
            else:  
                for shape in shapes:
                        point =  np.array(shape["points"] )       
                        contours = point.astype('int') 
                        cv2.drawContours(gray_im, [contours], -1,(255,255,255),-1)
                        poly = self.point_reshape(contours,o_image,[300,300],self.point_length)           
                        points.append(poly)  
      #  gray_filed= cv2.cvtColor(gray_im.astype('uint8'), cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('11.png',gray_im)
        # cv2.imwrite('2.png',image)      
        #0410
        angle_field = self.init_angle_field(points,[image.shape[0],image.shape[1]], line_width=1)


        grad_x = cv2.Sobel(angle_field, cv2.CV_32F, 1, 0)  # 对x求一阶导
        grad_y = cv2.Sobel(angle_field, cv2.CV_32F, 0, 1)
        
        #0415
        # grad_x = cv2.Sobel(gray_filed, cv2.CV_32F, 1, 0)  # 对x求一阶导
        # grad_y = cv2.Sobel(gray_filed, cv2.CV_32F, 0, 1)
     
        gradx = cv2.convertScaleAbs(grad_x)  
        grady = cv2.convertScaleAbs(grad_y)  
        filed = [gradx/255, grady/255]
        if self.transform is not None:
            image, label, filed = self.transform(image, label,filed)
         
#        points  = torch.from_numpy(np.array(points))
        if self.split=='test' or self.split=='val':
            return image, label, filed, point_path,name
        else:
            return image, label, filed, point_path
    
    def draw_circle(self,draw, center, radius, fill):
        draw.ellipse([center[0] - radius,
                  center[1] - radius,
                  center[0] + radius,
                  center[1] + radius], fill=fill, outline=None)
                  
    def draw_linear_ring(self,draw, linear_ring, line_width):
        # --- edges:
        coords = np.array(linear_ring)
        edge_vect_array = np.diff(coords, axis=0)
        edge_angle_array = np.angle(edge_vect_array[:, 1] + 1j * edge_vect_array[:, 0])  # ij coord sys
        neg_indices = np.where(edge_angle_array < 0)
        edge_angle_array[neg_indices] += np.pi

        first_uint8_angle = None
        for i in range(coords.shape[0] - 1):
            edge = (coords[i], coords[i + 1])
            angle = edge_angle_array[i]
            uint8_angle = int((255 * angle / np.pi).round())
            if first_uint8_angle is None:
                first_uint8_angle = uint8_angle
            line = [(edge[0][0], edge[0][1]), (edge[1][0], edge[1][1])]
            draw.line(line, fill=uint8_angle, width=line_width)
            self.draw_circle(draw, line[0], radius=line_width / 2, fill=uint8_angle)

        # Add first vertex back on top (equals to last vertex too):
        self.draw_circle(draw, line[1], radius=line_width / 2, fill=first_uint8_angle)
    
    def point_reshape(self,points,o_image,image,normallength):       
        points  = self.point_resize(points,o_image,image)
        return points
        # dp_threshold = 10
        # out = points
        # while(normallength!=len(points)):            
        #     if (len(points)< normallength):
        #         add = np.zeros([normallength-len(points),2])                   
        #         out = np.concatenate([out,add],axis=0)
        #         break
        #     elif normallength==len(points):
        #         out = points.reshape([-1,2])
        #         break
        #     else:
        #         dp_points = cv2.approxPolyDP(points, dp_threshold, True)
        #         points = dp_points
        #         dp_threshold = dp_threshold +5
        #         out = points.reshape([-1,2])
        # return out
    def point_resize(self,points,o_image,image):
    
        H,W,_ =o_image.shape
        o_H,o_W=image
        y_scale=o_H/H
        x_scale=o_W/W
        points[:,1]=(y_scale*points[:,1])
        points[:,0]=(x_scale*points[:,0])
        return points
    def init_angle_field(self,polygons, shape, line_width=1):

        assert type(polygons) == list, "polygons should be a list"
        polygons = [shapely.geometry.Polygon(a) for a in polygons]
        if len(polygons):
            assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

        im = Image.new("L", (shape[1], shape[0]))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)

        for polygon in polygons:
            self.draw_linear_ring(draw, polygon.exterior, line_width)
            for interior in polygon.interiors:
                self.draw_linear_ring(draw, interior, line_width)
        # Convert image to numpy array
        array = np.array(im)

        return array
