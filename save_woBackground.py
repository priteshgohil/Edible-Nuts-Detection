# This script reads the image from all the dataset folder, save rescaled image and label
import torch.utils.data as data
import numpy as np
import glob
import os
import sys
import torch
import json
import platform
import csv
from IPython.display import display

from PIL import Image
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


def make_dataset(path):
    #default "../data/"
    data = []
    for root,d_names,f_names in os.walk(path):
        for f in f_names:
            if('.png' in f):
                data.append(os.path.join(root,f))
    return data

def get_annotation(path):
    annotation = []
    with open(path) as f:
        objects = json.load(f)
    for obj in objects['shapes']:
        annotation.append([obj['label'] , np.array(obj['points'],dtype=np.uint)])
    return np.array(annotation)

def read_bb_annotation(csv_file):  
    with open(csv_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        labels = []
        for row in readCSV:
            labels.append(row)
        return np.array(labels[1:])

def get_polygon_mask(im, annotations):
    height,width,depth = im.shape
    mask = np.zeros(im.shape, np.uint8)
    ignore_mask_color = (255,)*depth
    for i,an in enumerate(annotations):
        polygon = an[1][np.newaxis]
        cv2.fillPoly(mask, polygon , ignore_mask_color)
        #u can remove below text later
#         cv2.putText(mask, an[0], (int(an[1][0,0]),int(an[1][0,1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    return mask

def mask_BW(image, mask):
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def scale_image(im, bounding_boxes, wsize=416):
    w_old = im.shape[1]
    h_old = im.shape[0]
    w_new = wsize
    h_new = 0
    
    #find ratio of new height and width
    wpercent = (w_new/float(w_old))
    h_new = int((float(h_old)*float(wpercent)))
    hpercent = (h_new/float(h_old))
    
    #resize image
    resized_im = cv2.resize(im,(w_new,h_new))
    
    #resize bounding box
    new_bb = []
    for bb in bounding_boxes:
        x,y,w,h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        x = int(x*float(wpercent))
        y = int(y*float(hpercent))
        w = int((float(w)*float(wpercent)))
        h = int((float(h)*float(hpercent)))
        new_bb.append([x,y,w,h])
    return resized_im, np.array(new_bb,dtype=bounding_boxes.dtype)

def display_img(im_path, an_path, count):
    im = cv2.imread(im_path)
    annotation = get_annotation(an_path)
    mask = get_polygon_mask(im,annotation)
    masked_image = mask_BW(im.copy(), mask)
    
    rows = np.where(bb_annotation==an_path.split('/')[-1])[0]
    bb = bb_annotation[rows,:]
    resized_im, new_bb = scale_image(masked_image.copy(), bb[:,1:5])
    if((count+1)%5==0):
        resized_im = cv2.resize(im,(416, 261))
        cv2.imwrite("/media/pritesh/Entertainment/yolo/PyTorch-YOLOv3/data/custom_im2/samples/{}".format(im_path.split("/")[-1]), resized_im)
    else:
        resized_im = cv2.resize(im,(416, 261))
        cv2.imwrite("/media/pritesh/Entertainment/yolo/PyTorch-YOLOv3/data/custom_im2/images/{}".format(im_path.split("/")[-1]), resized_im)
        write_file("/media/pritesh/Entertainment/yolo/PyTorch-YOLOv3/data/custom_im2/labels/{}".format(im_path.split("/")[-1].split(".")[0]), new_bb, bb[:,-1], resized_im.shape)

def write_file(path, bounding_box, label ,size):
    height,width = size[0], size[1]
    file1 = open(path+".txt","a") 
    # \n is placed to indicate EOL (End of Line)
    for i,bb in enumerate(bounding_box):
        x,y,w,h = float(bb[0])/float(width), float(bb[1])/float(height), \
            float(bb[2])/float(width), float(bb[3])/float(height)
        line = ["{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(label[i], x,y,w,h)]
        file1.writelines(line) 
    file1.close() #to change file access modes 


data = make_dataset("/media/pritesh/Entertainment/cvData/")
csv_file_path = "/media/pritesh/Entertainment/cvData/bb_annotation.csv"
labels = [x.replace('image', 'label_renamed').replace(os.path.splitext(x)[-1], '.json') for x in data]
bb_annotation = read_bb_annotation(csv_file_path)
count = 0

for i,im_path in enumerate(data):
    display_img(im_path, labels[i], i)
#    if(i==101):
#        break
