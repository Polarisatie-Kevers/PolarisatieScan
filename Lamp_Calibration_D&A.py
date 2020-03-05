# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:08:07 2020

@author: Daan Lytens
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from rawpy import imread
import os

#Het expliciete pad is hier naar de series van het lampje onder verschillende hoeken
path_calibr1 = r'C:/Users/Daan Lytens/Documents/UNI1920/BachelorProject/27-02-20_Lampje_onder_hoeken/Binnenste Cirkel/'
directory = r'C:/Users/Daan Lytens/Documents/UNI1920/BachelorProject/27-02-20_Lampje_onder_hoeken/Binnenste Cirkel/Results/'
path_calibr_darks = r'C:/Users/Daan Lytens/Documents/UNI1920/BachelorProject/27-02-20_Darks/'

path_calibr3 = r'C:/Users/Daan Lytens/Documents/UNI1920/BachelorProject/27-02-20_Lampje_onder_hoeken/Buitenste Cirkel/'

def filenames(path):
    names = []
    directory = os.listdir(path)
    for file in directory:
        if '.NEF' in file:
            names.append(str(file))
        else:
            continue
    return names

def mean_array(array, array2):
    mean = (array+array2)/2
    return mean


def read_data(directory, file):
    raw_data = imread(directory + file)
    raw_img = raw_data.raw_image.astype(np.float64)
    return raw_img

#We hebben 2 series van foto's zonder licht om de slechte pixels die nog intensiteit geven
#in kaart te brengen, dit gemiddelde kan dan van de gefilterde foto's afgetrokken worden.
#we verwachten dan een betere indicatie van 'pixelbelichting' te hebben dan 150 van de originele code
"""Geeft een 3D array mee waarin de volgorde van diepte is Qp, Qm,Up, Um, Vp, Vm, I_observed, Il0, Il1, Il2"""
def bad_pixels(directory):
    names = filenames(directory)
    array_filters1 = np.zeros((4016, 6016, 10))
    array_filters2 = np.zeros((4016, 6016, 10))
    for i in range(0,20):
        if i < 10:
            array_filters1[:,:,i] = read_data(directory, names[i])
        else:
            array_filters2[:,:,i - 10] = read_data(directory, names[i])
    mean = 0.5 * (array_filters1 + array_filters2)
    return mean


def find_lamp(grayscale_image):
    indices = np.zeros(3)
    for i in range(500, len(grayscale_image[:,0])):
        for j in range(0, len(grayscale_image[0,:])):
            if grayscale_image[i,j] != 0:
                new_row = np.array([i, j, grayscale_image[i,j]])
                indices = np.vstack([indices,new_row])
                print(new_row)
    indices = indices[1:]
    return indices


filescirkel1 = filenames(path_calibr1)
filescirkel3 = filenames(path_calibr3)

meandark = bad_pixels(path_calibr_darks)

raw_data_Qp = read_data(path_calibr3, filescirkel3[40])
data_Qp = raw_data_Qp[500:,:] - meandark[500:,:,0]
plt.imshow(data_Qp)
plt.colorbar()
plt.show()