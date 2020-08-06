# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 02:36:46 2016

@author: dirie
"""

import glob, os

from os import rename, listdir

badprefix = "black"
#fnames = listdir('../dataset/')
#print(fnames)

path = '/home/dirie/HonoursProject/PYTHON/vehicle_color/black/'
ext = '.jpg'

p = '/home/dirie/HonoursProject/PYTHON/vehicle_color/yellow/'



i=773
for f in glob.glob("/home/dirie/HonoursProject/PYTHON/vehicle_color/yellow/truck/*.jpeg"):
    os.rename(f,p + 'yellow' + str(i)+'.jpg')
    #print(f)
    i=i+1
    
