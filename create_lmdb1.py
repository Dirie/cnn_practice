# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 11:57:06 2016

@author: dirie
"""

import os
import sys
sys.path.append("/home/dirie/deep-learning/caffe/python")
import caffe

from caffe.proto import caffe_pb2
import lmdb

import glob
import numpy as np
import random

import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg
import cv2


""" the width and height of each image  """
w = 256
h = 256

class cat_dog(object):
    """
    this class will be implemented all staff in chapter 13.
    """

    def __init__(self,w,h,train_data,test_data):
        """Return a new Truck object."""
        print("Welcome to the all implementation of image processing functions.");
        self.w = w
        self.h = h
        self.train_data = train_data
        self.test_data = test_data

    def transform_img(self,img):
        """Return the whitening of an image."""
        print("whitening method:");
        I = img
        I[:,:,0] = cv2.equalizeHist(I[:,:,0]);
        I[:,:,1] = cv2.equalizeHist(I[:,:,1]);
        I[:,:,2] = cv2.equalizeHist(I[:,:,2]);
        I = cv2.resize(I,(self.w,self.h),interpolation = cv2.INTER_CUBIC)
        print(I.shape)
        return I
        

    def make_datam(self,img,label):
        return caffe_pb2.Datum(
        channels=3,
        width = self.w,
        height=self.h,
        label=label,
        data=np.rollaxis(img,2).tostring())
        
        
    def create_lmdb_train(self):
        print ('Creating train_lmdb')
        train_lmdb = path +'train_lmdb'
        train_txt = path + 'train_txt'
        os.system('rm -rf  ' + train_lmdb)
        
        in_db = lmdb.open(train_lmdb, map_size=int(1e12))
        #in_txt = os.open(train_txt, 'r+')
        with in_db.begin(write=True) as in_txn:
            print(len(train_data))
            for in_idx, img_path in enumerate(self.train_data):
                if in_idx %  6 == 0:
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = self.transform_img(img)
                if 'black' in img_path:
                    label = 0
                    print('black')
                elif 'green' in img_path:
                    label = 1
                    print('green')
                elif 'blue' in img_path:
                    label = 2
                    print('blue')
                elif 'red' in img_path:
                    label = 3
                    print('red')
                else:
                    label = 4
                    print('white')
                datum =self.make_datam(img,label)  
                in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
                #print ('%:0>5d'.format(in_idx) + ':' + img_path)
        in_db.close()


    def create_lmdb_test(self):
        print ('\nCreating validation_lmdb')
        validation_lmdb = path + 'validation_lmdb'
        os.system('rm -rf  ' + validation_lmdb)
        
        in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(self.test_data):
                if in_idx % 6 != 0:
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                #img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
                img = self.transform_img(img)
                print(img.shape)
                if 'black' in img_path:
                    label = 0
                elif 'green' in img_path:
                    label = 1
                elif 'blue' in img_path:
                    label = 2
                elif 'red' in img_path:
                    label = 3
                else:
                    label = 4
                datum =self.make_datam(img,label) 
                #datum = make_datum(img, label)
                in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
                #print ('%:0>5d'.format(in_idx) + ':' + img_path)
        in_db.close()






path = '/home/dirie/cnn_practice/input/'
#train_lmdb = '/home/dirie/test/input/train_lmdb'
#validation_lmdb = '/home/dirie/test/input/validation_lmdb'

#os.system('rm -rf  ' + train_lmdb)
#os.system('rm -rf  ' + validation_lmdb)


train_data = [img for img in glob.glob("/home/dirie/cnn_practice/input/train/*jpg")]
test_data = [img for img in glob.glob("/home/dirie/cnn_practice/input/train/*jpg")]



#Shuffle train_data
random.shuffle(train_data)

random.shuffle(test_data)

P = cat_dog(w,h,train_data,test_data)

P.create_lmdb_train()
P.create_lmdb_test()













#cap = cv2.VideoCapture('jan28.avi')
##P = cat_dog(frame,100,100)
#while(cap.isOpened()):
#    ret, frame = cap.read()
#
#    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    P = cat_dog(frame,250,400)
#    cv2.imshow('frame',P.transform_img())
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cap.release()
#cv2.destroyAllWindows()



