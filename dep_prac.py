# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 02:06:22 2016

@author: dirie
"""

import os
import sys
sys.path.append("/home/dirie/deep-learning/caffe/python")
import caffe
import numpy as np
import matplotlib.pyplot as plt
import cv2

caffe.set_mode_cpu()


#net = caffe.Net('/home/dirie/cnn_practice/myconvnet.prototxt', caffe.TEST)
net = caffe.Net('/home/dirie/cnn_practice/myconvnet.prototxt', caffe.TEST)


print (net.inputs)
print net.blobs['conv'].data.shape



im = cv2.imread('/home/dirie/deep-learning/caffe/examples/images/cat_gray.jpg',0)
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

net.forward()

for i in range(3):
    cv2.imwrite('frame'+ str(i) + '.jpg',net.blobs['conv'].data[0,i])

net.save('myconvmodel.caffemodel')