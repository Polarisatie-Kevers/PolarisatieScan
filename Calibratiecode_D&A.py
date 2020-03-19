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
import winsound

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
 
xmin = 2150
xmax = 3900
ymin = 760
ymax = 1470
x = np.arange(xmin, xmax)
y = np.arange(ymin, ymax)
x_small = np.arange(xmin/2, xmax/2)
y_small = np.arange(ymin/2, ymax/2)

def _find_offset(color_pattern, colour):
    pos = np.array(np.where(color_pattern == colour)).T[0]
    return pos


def demosaick(bayer_map, *data, **kwargs):
    """
    Simplified demosaicking method for RGBG data.
    Uses a Bayer map `bayer_map` (RGBG channel for each pixel) and any number
    of input arrays `data`. Any additional **kwargs are passed to pull_apart.
    """
    # Demosaick the data
    data_RGBG = [pull_apart(data_array, bayer_map, **kwargs)[0] for data_array in data]

    # If only a single array was given, don't return a list
    if len(data_RGBG) == 1:
        data_RGBG = data_RGBG[0]

    return data_RGBG


def pull_apart(raw_img, color_pattern, color_desc=b"RGBG"):
    if color_desc != b"RGBG":
        raise ValueError(f"Image is of type {raw_img.color_desc} instead of RGBG")
    offsets = np.array([_find_offset(color_pattern, i) for i in range(4)])
    offX, offY = offsets.T
    R, G, B, G2 = [raw_img[x::2, y::2] for x, y in zip(offX, offY)]
    RGBG = np.stack((R, G, B, G2))
    return RGBG, offsets

def put_together_from_offsets(R, G, B, G2, offsets):
    result = np.zeros((R.shape[0]*2, R.shape[1]*2))
    for colour, offset in zip([R,G,B,G2], offsets):
        x, y = offset
        result[x::2, y::2] = colour
    result = result.astype(R.dtype)
    return result


def put_together_from_colours(RGBG, colours):
    original = np.zeros((2*RGBG.shape[1], 2*RGBG.shape[2]))
    for j in range(4):
        original[np.where(colours == j)] = RGBG[j].ravel()
    return original


def split_RGBG(RGBG):
    R, G, B, G2 = RGBG
    return R, G, B, G2


def to_RGB_array(raw_image, color_pattern):
    RGB = np.zeros((*raw_image.shape, 3))
    R_ind = np.where(color_pattern == 0)
    G_ind = np.where((color_pattern == 1) | (color_pattern == 3))
    B_ind = np.where(color_pattern == 2)
    RGB[R_ind[0], R_ind[1], 0] = raw_image[R_ind]
    RGB[G_ind[0], G_ind[1], 1] = raw_image[G_ind]
    RGB[B_ind[0], B_ind[1], 2] = raw_image[B_ind]
    return RGB


def cut_out_spectrum(raw_image):
    cut = raw_image[ymin:ymax, xmin:xmax]
    return cut


def multiply_RGBG(data, colours, factors):
    data_new = data.copy()
    for j in range(4):
        data_new[colours == j] *= factors[j]
    return data_new

def save_file(directory,filename,image):
    cv2.imwrite(directory + filename, image)
    return

def normaliseren(rcp_lcp):
    rcp_lcp -= rcp_lcp.mean()
    rcp_lcp /= rcp_lcp.std()
    return rcp_lcp


def ab(image1, image2):
    diff = image1 - image2
    grady, gradx = np.gradient(image2)
#    ab = np.zeros(2)
    #som = np.array([0])
    a, b = 0, 0
    rangeab= np.linspace(-1, 1, 1000)
    func = 10000000000
    for i in rangeab:
        for j in rangeab:
            new_func = np.sum(( (diff - i * gradx - j * grady) ** 2))
            if new_func < func:
                func = new_func
                a, b = i, j
    frequency = 440
    duration = 1500  
    winsound.Beep(frequency, duration)
    return a, b, func

def displace(im1, im2, row_range_b, row_range_e, col_range_b, col_range_e):
    p = im1[row_range_b:row_range_e, col_range_b:col_range_e]
    m = im2[row_range_b:row_range_e, col_range_b:col_range_e]
    a, b, som = ab(p, m)
    grady, gradx = np.gradient(m)
    f = (p - m) -  (a) * gradx - (b) * grady
    g = np.sum((p - m)**2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(p - m)
    fig.colorbar(ax1.imshow(p - m), ax = ax1)
    ax1.set_title('(f1 - f2) = ' + str(round(g, 3)))
    ax2.imshow(f)
    fig.colorbar(ax2.imshow(f), ax = ax2)
    ax2.set_title('(f1 - f2) - a *grad(x) - b * grad(y) = '+ str(round(np.sqrt(som), 3))+ '\n(a,b) is (' + str(round(a, 3)) +' , '+ str(round(b, 3))+' )')
    return

#Het expliciete pad is hier naar de series die genomen zijn zonder verschuiving van de lades
path_calibr = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/03-03-2020/Voor Calibratie/Grid/'
directory = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/03-03-2020/Results Calibratie Grid/'

names_test = filenames(path_calibr)
#Inlezen van RHCP foto en LCHP foto, Foto zonder filter

#rcp1, lcp1, zonder_filter1 = [read_data(path_calibr + names_test[i]) for i in range(4,7)]
Qp, Qm = [(normaliseren(read_data(path_calibr + names_test[i]))) for i in range(0,2)]

#region: [3205:3500, 3348:3652] met 8 vierkantjes
#region: [2033:2183, 1260:1408] met 4 vierkantjes
#region: [1252:1332, 1599:1678] met 2 vierkantjes
#displace(Qp, Qm, 3205, 3500, 3348, 3652)
#displace(Qp, Qm, 2033, 2183, 1260, 1408)
#displace(Qp, Qm, 1252, 1332, 1599, 1678)

#Uit elkaar trekken in RGBG

pattern_Qp=imread(path_calibr + names_test[0]).raw_pattern
Qp_RGBG = demosaick(pattern_Qp, Qp)
pattern_Qm=imread(path_calibr + names_test[1]).raw_pattern
Qm_RGBG = demosaick(pattern_Qm, Qm)

#rcp1_R, rcp1_G, rcp1_B, rcp1_G2 = split_RGBG(rcp1_RGBG)
#lcp1_R, lcp1_G, lcp1_B, lcp1_G2 = split_RGBG(lcp1_RGBG)
#rcp_lcpR = normaliseren(rcp1_R - lcp1_R)
#rcp_lcpG = normaliseren(rcp1_G - lcp1_G)
#rcp_lcpB = normaliseren(rcp1_B - lcp1_B)
#rcp_lcpG2 = normaliseren(rcp1_G2 - lcp1_G2)
Qp_R, Qp_G, Qp_B, Qp_G2 = split_RGBG(Qp_RGBG)
Qm_R, Qm_G, Qm_B, Qm_G2 = split_RGBG(Qm_RGBG)
#Qp_QmR, Qp_QmG, Qp_QmB, Qp_QmG2 = normaliseren(Qp_R - Qm_R), normaliseren(Qp_G - Qm_G), normaliseren(Qp_B - Qm_B), normaliseren(Qp_G2 - Qm_G2)  
Qp_Rn, Qp_Gn, Qp_Bn, Qp_G2n = normaliseren(Qp_R), normaliseren(Qp_G), normaliseren(Qp_B), normaliseren(Qp_G2)  
Qm_Rn, Qm_Gn, Qm_Bn, Qm_G2n = normaliseren(Qm_R), normaliseren(Qm_G), normaliseren(Qm_B), normaliseren(Qm_G2)
#lcp1_R, lcp1_G, lcp1_B, lcp1_G2 = split_RGBG(lcp1_RGBG)
#rcp_lcpR, rcp_lcpG, rcp_lcpB, rcp_lcpG2 = normaliseren(rcp1_R - lcp1_R), normaliseren(rcp1_G - lcp1_G), normaliseren(rcp1_B - lcp1_B), normaliseren(rcp1_G2 - lcp1_G2)  

#displace(Qp_Gn, Qm_Gn, 1110, 1270, 900, 1070) # 8 cubes
#displace(Qp_Gn, Qm_Gn, 1150, 1230, 945, 1025) # 4 cubes
displace(Qp_Gn, Qm_Gn, 1170, 1215, 960, 1005) # 2 cubes

#plt.imshow(rcp1_RGBG[1,1200:1650,1480:1860], vmin=-1000, vmax=1000)
#plt.colorbar()

"""
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(rcp_lcpR[:,0:4850], vmin = -50, vmax = 50)
axs[0, 0].set_title('Red')
axs[1, 0].imshow(rcp_lcpG[:,0:4850], vmin = -50, vmax = 50)
axs[1, 0].set_title('Green')
axs[0, 1].imshow(rcp_lcpB[:,0:4850], vmin = -50, vmax = 50)
axs[0, 1].set_title('Blue')
axs[1, 1].imshow(rcp_lcpG2[:,0:4850], vmin = -50, vmax = 50)
axs[1, 1].set_title('Green2')
#plt.colorbar()
plt.show()


pattern_r2 = imread(path_calibr + names_test[14]).raw_pattern
rcp2_RGBG = demosaick(pattern_r2, rcp2)
lcp2_RGBG = demosaick(imread(path_calibr + names_test[15]).raw_pattern, lcp2)

rcp2_R, rcp2_G, rcp2_B, rcp2_G2 = split_RGBG(rcp2_RGBG)
lcp2_R, lcp2_G, lcp2_B, lcp2_G2 = split_RGBG(lcp2_RGBG)
rcp2_lcpR = normaliseren(rcp2_R - lcp2_R)
rcp2_lcpG = normaliseren(rcp2_G - lcp2_G)
rcp2_lcpB = normaliseren(rcp2_B - lcp2_B)
rcp2_lcpG2 = normaliseren(rcp2_G2 - lcp2_G2)

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(rcp2_lcpR[:,0:4850], vmin = -50, vmax = 50)
axs[0, 0].set_title('Red')
axs[1, 0].imshow(rcp2_lcpG[:,0:4850], vmin = -50, vmax = 50)
axs[1, 0].set_title('Green')
axs[0, 1].imshow(rcp2_lcpB[:,0:4850], vmin = -50, vmax = 50)
axs[0, 1].set_title('Blue')
axs[1, 1].imshow(rcp2_lcpG2[:,0:4850], vmin = -50, vmax = 50)
axs[1, 1].set_title('Green2')
#plt.colorbar()
plt.show()

pattern_r3 = imread(path_calibr + names_test[64]).raw_pattern
rcp3_RGBG = demosaick(pattern_r3, rcp3)
lcp3_RGBG = demosaick(imread(path_calibr + names_test[65]).raw_pattern, lcp3)

rcp3_R, rcp3_G, rcp3_B, rcp3_G2 = split_RGBG(rcp3_RGBG)
lcp3_R, lcp3_G, lcp3_B, lcp3_G2 = split_RGBG(lcp3_RGBG)
rcp3_lcpR = normaliseren(rcp3_R - lcp3_R)
rcp3_lcpG = normaliseren(rcp3_G - lcp3_G)
rcp3_lcpB = normaliseren(rcp3_B - lcp3_B)
rcp3_lcpG2 = normaliseren(rcp3_G2 - lcp3_G2)

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(rcp3_lcpR[0:,0:4850], vmin = -50, vmax = 50)
axs[0, 0].set_title('Red')
axs[1, 0].imshow(rcp3_lcpG[:,0:4850], vmin = -50, vmax = 50)
axs[1, 0].set_title('Green')
axs[0, 1].imshow(rcp3_lcpB[:,0:4850], vmin = -50, vmax = 50)
axs[0, 1].set_title('Blue')
axs[1, 1].imshow(rcp3_lcpG2[:,0:4850], vmin = -50, vmax = 50)
axs[1, 1].set_title('Green2')
#plt.colorbar()
plt.show()

plot.show_image_RGBG2(rcp3_RGBG-lcp3_RGBG)
#cv2.imwrite(directory + 'rcp_lcp.png', rcp_lcp1)
#rcp_lcp *= 200
#plt.imshow(rcp_lcp)
"""
