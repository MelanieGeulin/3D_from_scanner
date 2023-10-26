#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:31:46 2021

@author: kuriozity
"""

import cv2
from skimage import io,morphology,measure
import matplotlib.pyplot as plt
import numpy as np
import os

def trim_(im,min_x,max_x,min_y,max_y):
    im_trimed = im[min_y:max_y,min_x:max_x]
    return im_trimed

def histogram_(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.show()

#Load IRM images 
def load_images_from_folder(folder):
    #Declare variables
    images=[]
    files = []
    
    #Retreives folder's images datas
    for filename in os.listdir(folder):
        files.append(filename)
    files = sorted(files)
    for f in files:
        img = cv2.imread(os.path.join(folder,f))  
        if img is not None:
            images.append(img)
    return images

#Show image result
def image_show(im):
    plt.figure()
    plt.imshow(im)
    plt.show()

#Image Treatment
def tempered_(im,nosection,section):
    #Declare variables
    end_liver = 0
    plus = 0
    k = 0
    
    if im.ndim != 2:
        #From 3dim RGB to 2dim grayscale
        im_grey = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    else:
        im_grey = im
    if nosection == 1:
        im_trimed = trim_(im_grey,0,np.int64(im.shape[1]-im.shape[1]/3),800,1800)
    else:
        im_trimed = trim_(im,section[2][0],section[3][0],section[0][0],section[1][0])
    
    #Binarization with high threshold for bones
    th,im_all = cv2.threshold(im_trimed,80,255,cv2.THRESH_BINARY)
    #Binarization with mild threshold for contrast organs
    th,im_os = cv2.threshold(im_trimed,175,255,cv2.THRESH_BINARY)
    #Binarized image without bones
    im_org = im_all-im_os
    #Take off small objects
    erosion = morphology.binary_erosion(im_org,morphology.disk(20))
    erosion_bones = morphology.binary_erosion(im_os,morphology.disk(20))
    #Fill small holes
    dilation = morphology.binary_dilation(erosion,morphology.disk(15))
    dilation_bones = morphology.binary_dilation(erosion_bones,morphology.disk(15))
    #Separate different region
    labels = morphology.label(dilation,background = 0)
    labels_bones = morphology.label(dilation_bones,background = 0)
    #Contour detection
    contours = measure.find_contours(labels,0.5)
    contours_bones = measure.find_contours(labels_bones,0.5)
    
    # Show starting image
    # Take only biggest area contoured
    while k != 1:
        k = 0
        stains = []
        for i in range(len(contours)):
            if contours[i].shape[0] > 1+plus:
                stains.append(contours[i]) 
                k = k+1
        if k == 0:
            end_liver = 1
            k = 1
        plus = plus+10
    
    stains_bones = []
    for i in range(len(contours_bones)):
        stains_bones.append(contours_bones[i])
        
    #Show localized organ on starting image         
    # for contour in stains:
    #     image_show(im)
    #     plt.plot(contour[:,1],contour[:,0],linewidth=1,c='r')        
    # plt.show()   
     
    return stains,stains_bones,im_trimed

#Show 3d contouring result of localized organ
def contours_3d(stain,i):
    k = i-1
    plt.figure()
    ax = plt.axes(projection='3d')
    for j in range(i):
        ax.plot3D(stain[j][0][:,1],stain[j][0][:,0],k/1000,linewidth = 1, c = 'g')
        k = k-1

#Defining section of localized organs
def section_(stain,i):
    #Declare variables
    section = []
    section_max = np.zeros((4,1))
    
    #Find coordinates of squares
    for k in range(i):
        max_x = np.max(stain[k][0][:,0])-50
        max_y = np.max(stain[k][0][:,1])+50
        min_x = np.min(stain[k][0][:,0])-50
        min_y = np.min(stain[k][0][:,1])+50
        section.append(np.array([min_x,max_x,min_y,max_y]))
        
    section = np.array(section)
    section_max[0] = np.mean(section[:,0])
    section_max[1] = np.mean(section[:,1])
    section_max[2] = np.mean(section[:,2])
    section_max[3] = np.mean(section[:,3])

    return np.int64(section_max)
