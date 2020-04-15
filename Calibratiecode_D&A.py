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
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from scipy.signal import general_gaussian


#filenames creates a list of names of all .NEF files in a certain folder. If non .NEF files are
#in the folder, it doesn't append the object
def filenames(path):
    names = []
    directory = os.listdir(path)
    for file in directory:
        if '.NEF' in file:
            names.append(str(file))
        else:
            continue
    return names

#Select_every_nth creates a list of names of photos taken above the same polarizing filter
#since one serie of photos has 10 images it takes every 10th item within the nameslist created
#by the function 'filenames'
def select_every_nth(stringlist, beginvalue):
    lst = []
    for i in range(beginvalue, len(stringlist)+1, 10):
        if i < len(stringlist):
            lst.append(stringlist[i])
    return lst

#Read_data extracts raw data information from an image
def read_data(filename):
    raw_data = imread(str(filename))
    raw_img = raw_data.raw_image.astype(np.float64)
    raw_img -= 150 #average amount of background noise
    return raw_img   
 
    
#Line 55  - line 119 is the SPECTACLE code of Olivier
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

#Save_file saves an image to a certain directory folder
def save_file(directory,filename,image):
    cv2.imwrite(directory + filename, image)
    return

#Normalising: I_norm = [(I+) - (I-)]/[(I+) + (I-)]
def normaliseren(im1, im2):
    I = im1 + im2
    Inorm = im1/I
    return Inorm

#func is the assumed function for which the grid is displaced, necessary for finding the 
#value of (a,b) in the minimize_ab function below
def func(data, a, b):
    x, y = data
    return (a * x + b * y)
def minimize_ab(im1, im2):
    diff = np.ravel(im1 - im2) #Creates a vector of the matrix by adding each row behind eachother
    grady, gradx = np.gradient(im2)
    gradx_v = np.ravel(gradx)
    grady_v = np.ravel(grady)
    guess = np.array([0., 0.])
    ab, covar = curve_fit(func, (gradx_v, grady_v), diff, guess) #gives the best fitted values for a,b by using the ls-method
    return ab, covar, gradx, grady

#ab gives for two raw data images the best fitted vallues for a and b for 2x2 squares
#it returns rangex, rangey, gradx and grady aswell for further use in the shift function
def ab(im1, im2):
    rangex = np.arange(0, len(im1[0,:]), 17)
    rangey = np.arange(0, len(im1[:,0]), 17)
    a_array = np.zeros((len(rangey), len(rangex)))
    b_array = np.zeros((len(rangey), len(rangex)))
    for i in range(0, len(rangex)-2):
        #Taking the upper left corner to find the a,b values for 2x2 areas
        #The point that will use this a,b is located at i+1, j+1
        for j in range(0, len(rangey)-2):
            ab, cov, gradx, grady = minimize_ab(im1[rangey[j]:rangey[j+2],rangex[i]:rangex[i+2]], im2[rangey[j]:rangey[j+2],rangex[i]:rangex[i+2]])
            a_array[j,i] = ab[0]
            b_array[j,i] = ab[1]
    return a_array[0:int(len(a_array[:,0]) - 2),0:int(len(a_array[0,:]) - 2)], b_array[0:int(len(a_array[:,0]) - 2),0:int(len(a_array[0,:]) - 2)], rangex, rangey

def mask(im, centerx, centery):
    ystart, yend = (centery - 9), (centery + 8)
    xstart, xend = (centerx - 9), (centerx + 8)
    area = im[ystart:yend, xstart:xend]
    return area

#Shift makes use of the ab function in order to shift the center of each 2x2 squars by a certain amount
#im1 and im2 are the ones we calculate (a,b) for, im3 and im 4 are the images we want to shift on wrt eachother
def shift(im1, im2, im3, im4, row_b, row_e, col_b, col_e):
    im1 = im1[int(row_b):int(row_e), int(col_b):int(col_e)]
    im2 = im2[int(row_b):int(row_e), int(col_b):int(col_e)]
    im3 = im3[int(row_b):int(row_e), int(col_b):int(col_e)]
    im4 = im4[int(row_b):int(row_e), int(col_b):int(col_e)]
    shape = np.shape(im2)
    shifted = np.zeros(shape)
    grady, gradx = np.gradient(im4)
    a, b, rangex, rangey = ab(im1, im2)
    centerx, centery = rangex[1:], rangey[1:]
    for i in range(0, len(rangex)-2):
        for j in range(0, len(rangey)-2):
            #We now take the center of 2x2 squares in length of one square, in the center
            area1 = mask(im3, centerx[i], centery[j])
            area2 = mask(im4, centerx[i], centery[j])
            gradx1 = mask(gradx, centerx[i], centery[j])
            grady1 = mask(grady, centerx[i], centery[j])
            shifted[int(centery[j] - 9):int(centery[j]+8), int(centerx[i]-9):int(centerx[i]+8)] = (area1 - area2) -  (a[j, i]) * gradx1 - (b[j,i]) * grady1
    return shifted

#subplot_norm makes a plot of two plots, both with the same colorbar-values
#the left plot shows the resulting image without gradient adjustment
#im3 should be the shifted image, done by the shift function above
def subplot_norm(im1, im2, im3, title):
    diff = (im1 - im2)
    sumdiff, avgdiff, avgabsdiff, vardiff = np.sum(abs(diff)), np.mean(diff), np.mean(abs(diff)), np.var(diff)
    sumim3, avgim3, avgabsim3, varim3  = np.sum(abs(im3)), np.mean(im3), np.mean(abs(im3)), np.var(im3)
    print('total sum of ', title, ' is ', sumdiff)
    print('average value of ', title, ' is ', avgdiff)
    print('average value of abs value of ', title, ' is ', avgabsdiff)
    print('variance of ', title, ' is ', vardiff)
    print('total sum of ', title, ' with gradient is ', sumim3)
    print('average value of ', title, ' with gradient is ', avgim3)
    print('average value of abs value of ', title, ' with gradient is ', avgabsim3)
    print('variance of ', title, ' with gradient is ', varim3)
    print('diff_grad = ', (sumim3/sumdiff), '% diff in intensity')
    fig = plt.figure()
    ax = fig.add_subplot(121, autoscale_on='FALSE')
    ax.set_title(title)
    mesh = ax.pcolormesh(diff)
    mesh.set_clim(-1,1)
    ax1 = fig.add_subplot(122, autoscale_on='FALSE')
    ax1.set_title(title + 'adjusted with gradient')
    mesh1 = ax1.pcolormesh(im3)
    mesh1.set_clim(-1, 1)
    fig.colorbar(mesh,ax=ax, orientation ='horizontal')
    fig.colorbar(mesh1,ax=ax1,  orientation ='horizontal')
    #fig.tight_layout()
    plt.show()
    return

#ab_stats first makes the matrix of a- and b-values a vector and then places them
#under eachother in seperate rows. This way all values for a certain pixel in the 
#a or b matrix get arranged in a column. This is viable since all matrices have 
#the same shape. This function is then used to get statistical interpretations on the a an b values
def ab_stats(a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7, a8, b8, a9, b9, a10, b10, filtername):
    #matrixA and matrixB are both of shape (10, 12200) in the hard-coded case
    #aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8, aa9, aa10 = np.ravel(a1), np.ravel(a2), np.ravel(a3), np.ravel(a4), np.ravel(a5), np.ravel(a6), np.ravel(a7), np.ravel(a8), np.ravel(a9), np.ravel(a10) 
    matrixA = np.array((np.ravel(a1), np.ravel(a2), np.ravel(a3), np.ravel(a4), np.ravel(a5), np.ravel(a6), np.ravel(a7), np.ravel(a8), np.ravel(a9), np.ravel(a10)))
    #bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10 = np.ravel(b1), np.ravel(b2), np.ravel(b3), np.ravel(b4), np.ravel(b5), np.ravel(b6), np.ravel(b7), np.ravel(b8), np.ravel(b9), np.ravel(b10) 
    matrixB = np.array((np.ravel(b1), np.ravel(b2), np.ravel(b3), np.ravel(b4), np.ravel(b5), np.ravel(b6), np.ravel(b7), np.ravel(b8), np.ravel(b9), np.ravel(b10)))
    statsA = np.array(['min', 'max', 'mean', 'median', 'variance'])
    statsB = np.array(['min', 'max', 'mean', 'median', 'variance'])
    for i in range(len(matrixA[0,:])):
        minA, maxA, meanA, medianA, varA = np.min(matrixA[:,i]), np.max(matrixA[:,i]), np.mean(matrixA[:,i]), np.median(matrixA[:,i]), np.var(matrixA[:,i])
        addvectorA = np.array([minA, maxA, meanA, medianA, varA])
        statsA = np.vstack((statsA, addvectorA))
    for j in range(len(matrixB[0,:])):
        minB, maxB, meanB, medianB, varB = np.min(matrixB[:,j]), np.max(matrixB[:,j]), np.mean(matrixB[:,j]), np.median(matrixB[:,j]), np.var(matrixB[:,j])
        addvectorB = np.array([minB, maxB, meanB, medianB, varB])
        statsB = np.vstack((statsB, addvectorB))
    varA = statsA[1:, 4]
    varB = statsB[1:, 4]
    varA_pix = np.reshape(varA, (np.shape(a1))).astype(np.float)
    varB_pix = np.reshape(varB, (np.shape(b1))).astype(np.float)
    return statsA, statsB, matrixA, matrixB, varA_pix, varB_pix

#main function is where we calculate the optimized values of a and b for the RGBG images
def main(path, serie_b, serie_e, row_b, row_e, col_b, col_e):
    #Qpnames = select_every_nth(names_test, 0)
    #Qmnames = select_every_nth(names_test, 1)
    names = filenames(path) #All .NEF files in path_calibr
    Qp, Qm = [(read_data(path + names[i])) for i in range(serie_b, serie_e)]
#deviding the colors
    pattern_Qp=imread(path + names[serie_b]).raw_pattern
    Qp_RGBG = demosaick(pattern_Qp, Qp)
    pattern_Qm = imread(path + names[int(serie_e-1)]).raw_pattern
    Qm_RGBG = demosaick(pattern_Qm, Qm)
    Qp_R, Qp_G, Qp_B, Qp_G2 = split_RGBG(Qp_RGBG)
    Qm_R, Qm_G, Qm_B, Qm_G2 = split_RGBG(Qm_RGBG)
    Qp_Rn, Qp_Gn, Qp_Bn, Qp_G2n = normaliseren(Qp_R, Qm_R), normaliseren(Qp_G, Qm_G), normaliseren(Qp_B, Qm_B), normaliseren(Qp_G2, Qm_G2)
    Qm_Rn, Qm_Gn, Qm_Bn, Qm_G2n = normaliseren(Qm_R, Qp_R), normaliseren(Qm_G, Qp_G), normaliseren(Qm_B, Qp_B), normaliseren(Qm_G2, Qp_G2)
    #calculating a and b for different colors
    aR, bR, rangex, rangey = ab(Qp_Rn[row_b:row_e, col_b:col_e], Qm_Rn[row_b:row_e, col_b:col_e])
    aG, bG, rangex, rangey = ab(Qp_Gn[row_b:row_e, col_b:col_e], Qm_Gn[row_b:row_e, col_b:col_e])
    aB, bB, rangex, rangey = ab(Qp_Bn[row_b:row_e, col_b:col_e], Qm_Bn[row_b:row_e, col_b:col_e])
    aG2, bG2, rangex, rangey = ab(Qp_G2n[row_b:row_e, col_b:col_e], Qm_G2n[row_b:row_e, col_b:col_e])
#Shifting back  
    winsound.Beep(440, 300)
    winsound.Beep(300, 500)
    return aR, bR, aG, bG, aB, bB, aG2, bG2, Qp_Gn, Qm_Gn

#Total grid hardcoded: Image[238:1972, 323:2431] (devisible by 17)
#selected region of grid: Image[697:1411, 1088:1853] (devisible by 17)

#Explicit path to files of images
path = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/03-03-2020/Voor Calibratie/Grid/'
directory = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/03-03-2020/Results Calibratie Grid/'

aR1, bR1, aG1, bG1, aB1, bB1, aG21, bG21, Qp_Gn1, Qm_Gn1 = main(path, 0, 2, 238, 1972, 323, 2431)
aR2, bR2, aG2, bG2, aB2, bB2, aG22, bG22, Qp_Gn2, Qm_Gn2 = main(path, 10, 12, 238, 1972, 323, 2431)
aR3, bR3, aG3, bG3, aB3, bB3, aG23, bG23, Qp_Gn3, Qm_Gn3 = main(path, 20, 22, 238, 1972, 323, 2431)
aR4, bR4, aG4, bG4, aB4, bB4, aG24, bG24, Qp_Gn4, Qm_Gn4 = main(path, 30, 32, 238, 1972, 323, 2431)
aR5, bR5, aG5, bG5, aB5, bB5, aG25, bG25, Qp_Gn5, Qm_Gn5 = main(path, 40, 42, 238, 1972, 323, 2431)
aR6, bR6, aG6, bG6, aB6, bB6, aG26, bG26, Qp_Gn6, Qm_Gn6 = main(path, 50, 52, 238, 1972, 323, 2431)
aR7, bR7, aG7, bG7, aB7, bB7, aG27, bG27, Qp_Gn7, Qm_Gn7 = main(path, 60, 62, 238, 1972, 323, 2431)
aR8, bR8, aG8, bG8, aB8, bB8, aG28, bG28, Qp_Gn8, Qm_Gn8 = main(path, 70, 72, 238, 1972, 323, 2431)
aR9, bR9, aG9, bG9, aB9, bB9, aG29, bG29, Qp_Gn9, Qm_Gn9 = main(path, 80, 82, 238, 1972, 323, 2431)
aR10, bR10, aG10, bG10, aB10, bB10, aG210, bG210, Qp_Gn10, Qm_Gn10 = main(path, 90, 92, 238, 1972, 323, 2431)

#shiftQ1 = shift(Qp_Gn1, Qm_Gn1, Qp_Gn1, Qm_Gn1, 238, 1972, 323, 2431)
#subplot_norm(Qp_Gn1[238:1972, 323:2431], Qm_Gn1[238:1972, 323:2431], shiftQ1, 'Qp - Qm, Green, ')
#statsAG, statsBG, mA, mB, varApix, varBpix = ab_stats(aG1, bG1, aG2, bG2, aG3, bG3, aG4, bG4, aG5, bG5, aG6, bG6, aG7, bG7, aG8, bG8, aG9, bG9, aG10, bG10, 'Green')

diff = Qp_Gn1[238:1972, 323:2431] - Qm_Gn1[238:1972, 323:2431]

'''
window = np.outer(general_gaussian(diff.shape[1],1,1000, sym = 'False'),general_gaussian(diff.shape[1],1,1000,sym = 'False'))
rows = np.zeros((500, 3008))
diff = np.vstack((rows, diff))
diff = np.vstack((diff, rows))
diffw = np.dot(diff,window)

fta = np.fft.fft2(diffw)
ftashift = np.fft.fftshift(fta)
ftaabs = np.abs(ftashift)

plt.imshow(aG)# norm=LogNorm(vmin = 0.0001))
plt.colorbar()
'''
'''
ftb1 = np.fft.fft2(bG1)
ftb2 = np.fft.fft2(bG2)
#ftb3 = np.fft.fft2(bG3)
#ftb4 = np.fft.fft2(bG4)
#ftb5 = np.fft.fft2(bG5)
#ftb6 = np.fft.fft2(bG6)
#ftb7 = np.fft.fft2(bG7)
#ftb8 = np.fft.fft2(bG8)
#ftb9 = np.fft.fft2(bG9)
#ftbdiff = ftb1 - ftb2# (ftb1-ftb3)- np.abs(ftb1-ftb4)- np.abs(ftb1-ftb5)- np.abs(ftb1-ftb6) - np.abs(ftb1-ftb7)

ftbshift = np.fft.fftshift(ftbdiff)
ftbabs = np.abs(ftbshift)

plt.imshow(ftbabs, norm=LogNorm(vmin = 0.0001))
plt.colorbar()
'''
'''
fig = plt.figure()
ax = fig.add_subplot(121, autoscale_on='FALSE')
ax.set_title('Variance of a-values over 10 series (Green)')
mesh = ax.pcolormesh(varApix)
mesh.set_clim(0,0.005)
ax1 = fig.add_subplot(122, autoscale_on='FALSE')
ax1.set_title('Variance of b-values over 10 series (Green)')
mesh1 = ax1.pcolormesh(varBpix)
mesh1.set_clim(0, 0.1)
fig.colorbar(mesh,ax=ax, orientation ='horizontal')
fig.colorbar(mesh1,ax=ax1,  orientation ='horizontal')
#fig.tight_layout()
plt.show()'''

#displace was a function that used a 'manual' way of chosing a and b, not very efficient but 
#made some good first results
"""Selected region of the grid"""
#displace(Qp_Gn, Qm_Gn, 1110, 1270, 900, 1070) # 8 cubes
#displace(Qp_Gn, Qm_Gn, 1150, 1230, 945, 1025) # 4 cubes
#displace(Qp_Gn, Qm_Gn, 1170, 1215, 960, 1005) # 2 cubes

"""One square to the right wrt the figure above; One square is approx 41 pixels long"""
#displace(Qp_Gn, Qm_Gn, 1110, 1270, 941, 1111) # 8 cubes
#displace(Qp_Gn, Qm_Gn, 1150, 1230, 986, 1066) # 4 cubes
#displace(Qp_Gn, Qm_Gn, 1170, 1215, 1001, 1046) # 2 cubes

"""Bottom left corner on grid; total other region than above"""
#displace(Qp_Gn, Qm_Gn, 1669, 1745, 1969, 2049) # 4 cubes
#displace(Qp_Rn, Qm_Rn, 1669, 1745, 1969, 2049) # 4 cubes
#displace(Qp_Bn, Qm_Bn, 1669, 1745, 1969, 2049) # 4 cubes
#displace(Qp_G2n, Qm_G2n, 1669, 1745, 1969, 2049) # 4 cubes