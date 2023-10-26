#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:56:51 2021

@author: kuriozity
"""

from functions import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    folder = r'C:\Users\Esmeyralda Celiandre\Desktop\Programmes\3D_from_scanner\IRM_cut'
    images = load_images_from_folder(folder)
    
    stain = []
    stain_b = []
    i = 0 
    while i < len(images):
            print('Process Image Number: ', i+1)
            temp_liver,temp_bones,images[i] = tempered_(images[i],1,0)
            stain.append(temp_liver)
            stain_b.append(temp_bones)
            i = i+1
            print(np.shape(temp_liver))
    
    section_max = section_(stain,i)
    
    i = 0
    while i < len(images):
            print('Process Image Number: ', i+1)
            temp_liver,temp_bones,images[i] = tempered_(images[i],0,section_max)
            stain.append(temp_liver)
            stain_b.append(temp_bones)
            i = i+1
            

    stain = np.array(stain)
    stain_b = np.array(stain_b)

    contours_3d(stain,i)
    contours_3d(stain_b,i)