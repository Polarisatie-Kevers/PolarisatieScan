# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:49:44 2020

@author: Daan Lytens
"""
import numpy as np
from matplotlib import pyplot as plt
from rawpy import imread
from sys import argv, exc_info
from pathlib import Path
from spectacle import raw, plot
from spectacle.general import gaussMd, gauss_nan
from matplotlib.colors import Normalize, LightSource
import tkinter as tk
from tkinter import filedialog
import cv2
import os
import time  
from PIL import Image

def filenames(path):
    names = []
    directory = os.listdir(path)
    for file in directory:
        if '.NEF' in file:
            names.append(str(file))
        else:
            continue
    return names

def read_data(filename):
    raw_data = imread(str(filename))
    raw_img = raw_data.raw_image.astype(np.float64)
    raw_img -= 150
    return raw_img    

def save_file(directory,filename,image):
    cv2.imwrite(directory + filename, image)
    return


#Het expliciete pad is hier naar de series die genomen zijn zonder verschuiving van de lades
path_calibr = r'C:/Users/Daan Lytens/Documents/UNI1920/BachelorProject/03-03-20/03-03-20_Voor_Calibratie/Grid/'
directory = r'C:/Users/Daan Lytens/Documents/UNI1920/BachelorProject/03-03-20/03-03-20_Voor_Calibratie/Grid Results/'

names_test = filenames(path_calibr)
#Inlezen van RHCP foto en LCHP foto, Foto zonder filter

rcp1, lcp1, zonder_filter1 = [read_data(path_calibr + names_test[i]) for i in range(4,7)]
rcp2, lcp2, zonder_filter2 = [read_data(path_calibr + names_test[i]) for i in range(14,17)]
rcp3, lcp3, zonder_filter3 = [read_data(path_calibr + names_test[i]) for i in range(24,27)]

#Verschil tussen twee filters
zonder_rcp_1 = (0.5)*zonder_filter1 - rcp1 #In theorie blokt een circulaire filter 50% van het inkomende licht
zonder_lcp_1 = (0.5)*zonder_filter1 - lcp1 #In theorie blokt een circulaire filter 50% van het inkomende licht
rcp_rcp = rcp1 - rcp2

zonder_rcp_2 = (0.5)*zonder_filter2 - rcp2 #In theorie blokt een circulaire filter 50% van het inkomende licht
zonder_lcp_2 = (0.5)*zonder_filter2 - lcp2 #In theorie blokt een circulaire filter 50% van het inkomende licht
rcp_lcp= rcp2 - lcp2
lcp_lcp = lcp1 - lcp2

#normaliseren
rcp_lcp -= rcp_lcp.mean()
rcp_lcp /= rcp_lcp.std()
#rcp_lcp *= 200
#plt.imshow(rcp_lcp)

#plt.show()
#gradient_x = np.gradient(rcp_lcp, axis = )
laplacian = cv2.Laplacian(rcp_lcp,cv2.CV_64F)
sobelx = cv2.Sobel(rcp_lcp,cv2.CV_64F,1,0)
sobely = cv2.Sobel(rcp_lcp,cv2.CV_64F,0,1)
"""
plt.subplot(2,1,1),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.colorbar()
plt.subplot(2,1,2),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.colorbar()
#plt.savefig(directory+"Div_grid2")"""
plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.colorbar()
plt.show()
#Normaliseren van intensiteitµ: V_norm = (Vp - Vm)/(Vp + Vm)
#total_intensity_1 = rcp1 + lcp1
#normal_intensity_1 = rcp_lcp_1 / total_intensity_1


#cv2.imwrite(directory + 'total_img_rcplcp_rclcp2_zonderR_zonderL_verbetert.png', finalImage)
#cv2.imwrite(directory + 'rcp_lcp.png', rcp_lcp)

#Gradient berekenen van afbeelding/foto