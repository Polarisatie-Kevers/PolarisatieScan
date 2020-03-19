# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:31:46 2020

@author: Daan Lytens
"""
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import imageio
import elasticdeform
import matplotlib.pyplot as plt

def function(matrix):
    for i in range(0,len(matrix[0,:])):
        for j in range(0, len(matrix[:,0])):
            f = matrix[j,i] + 3*matrix[j,i] + 5
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
