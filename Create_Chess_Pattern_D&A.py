# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:31:46 2020

@author: Daan Lytens
"""
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import fftpack
from matplotlib.colors import LogNorm


def normaliseren(matrix):
    matrix -= matrix.mean()
    matrix /= matrix.std()
    return matrix
def func(data, a, b, c):
    x,y = data
    return (a*x + b*y + c)
def minimize_ab(im1, im2):
    diff = np.ravel(im1 - im2) #Creates a vector of the matrix by adding each row behind eachother
    grady, gradx = np.gradient(im2)
    gradx_v = np.ravel(gradx)
    grady_v = np.ravel(grady)
    guess = np.array([1.5, 2.,0])
    ab, covar = curve_fit(func, (gradx_v, grady_v), diff, guess) #gives the best fitted values for a,b by using the ls-method
    return ab, covar

def function(matrix):
    grady, gradx = np.gradient(matrix)
    for i in range(0,len(matrix[0,:])):
        for j in range(0, len(matrix[:,0])):
            f = gradx[j,i] + 3*grady[j,i] + 5
            matrix[j,i] = f
    return matrix

def chess_grid(numberx, numbery, length): #number: number of squares, length: how long are the squares
    board = np.zeros((numbery*length, numberx*length))
    for i in range(0, numbery*length, 2*length):
        for j in range(0, numberx*length, 2*length):
            board[j:(j+length), i:(i+length)] = 1
    for i in range(length, numbery*length, 2*length):
        for j in range(length, numberx*length, 2*length):
            board[j:(j+length), i:(i+length)] = 1
    return board

def chess_grid_deformed(numberx, numbery, length): #number: number of squares, length: how long are the squares
    board = np.zeros((numbery*length, numberx*length))
    for i in range(0, numbery*length, 2*length):
        for j in range(0, numberx*length, 2*length):
            board[j:(j+length), i:(i+length)] = 1
    for i in range(length, numbery*length, 2*length):
        for j in range(length, numberx*length, 2*length):
            board[j:(j+length), i:(i+length)] = 1
    board = np.roll(board, 2, 0)
    board = np.roll(board, 2, 1)
    return board

def sin(matrix):
    shape = np.shape(matrix)
    x0 = np.arange(shape[1])
    indicesx = np.zeros(shape)
    for i in range(0, shape[0]):
        row = x0 + i
        indicesx[i,:] = row
    sinM = np.sin(indicesx)
    return sinM

def ft_set0(im, vmin):
    for i in range(0, len(im[0,:])):
        for j in range(0, len(im[:,0])):
            if im[j,i] > vmin:
                im[j,i] = 0
    return im

#sinus = sin(grid)
#gridsin = np.dot(sinus, grid) #image
#ftsin = np.fft.fft2(gridsin) #fourier transform
#ftabs = np.abs(ftsin) #absolute value
#ftabs = np.fft.fftshift(ftabs) #shift

#ishifted = np.fft.ifftshift(ftabs)
#invft = np.fft.ifft2(ishifted)
#invftabs = np.abs(invft)

x = np.arange(1000)
y = np.arange(1000)
vx, vy = np.meshgrid(x, y)
vxsin = np.dot(np.sin(vx), vx)
sin = np.dot(vxsin, vx)

ft = np.fft.fft2(sin)
ftabs = np.abs(ft)
ftabs = np.fft.fftshift(ftabs)

plt.imshow(ftabs)
plt.colorbar()

ft0 = ft_set0(ftabs, 0)
ishifted = np.fft.ifftshift(ft0)
invft = np.fft.ifft2(ishifted)
invftabs = np.abs(invft)

