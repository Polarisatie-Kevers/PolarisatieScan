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
import math

#Het expliciete pad is hier naar de series van het lampje onder verschillende hoeken
path_calibr1 = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/27-02-20/Lampje onder hoeken/Binnenste Cirkel/'
path_calibr2 = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelorv Project/Images Scan/27-02-20/Lampje onder hoeken/Middelste Cirkel/'
path_calibr3 = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/27-02-20/Lampje onder hoeken/Buitenste Cirkel/'
directory = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/27-02-20/Lampje onder hoeken//Binnenste Cirkel/Results/'
result_lamp = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/27-02-20/Lampje onder hoeken/Results/'
path_calibr_darks = r'C:/Users/Daan Lytens/Documents/UNI1920/Bachelor Project/Images Scan/27-02-20/Darks/'

#Haalt de .NEF-files uit een map van de directory
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


"""Zoekt eerst naar de pixels met maximale intensiteit, daar zou de lamp zitten"""
def find_lamp_intensity(image):
    indices = np.zeros(2) #creeer een (0,0) vector om de indices op te slaan voor pixels waar de intensiteit maximaal is
    maxvalue = np.max(image) #hoogste pixelwaarde in de image
    for i in range(500, len(image[:,0])): #zoeken naar de juiste rij; begin bij 500 om de spiegels af te snijden
        for j in range(0, len(image[0,:])): #zoeken naar juiste kolom; begin bij 500 om spiegels af te snijden
            if image[i,j] == maxvalue:
                newrow = np.array([i,j])
                indices = np.vstack([indices, newrow]) #voeg nieuwe rij toe aan de vector
    indices = indices[1:,:] #gooi de nulde rij weg (dit is de [0,0])
    length = len(indices[:,0])
    if length % 2 != 0: #als de lengte geen even getal is, neem de middelste rij
        row = int(length/2) + (length % 2 > 0)
        middlerowvalues = indices[row,:]
    else:
        row = int(length/2)
        middlerowvalues = (indices[row,:])
    startcol = middlerowvalues[1] - 300
    endcol = middlerowvalues[1] + 300
    startrow = middlerowvalues[0] - 300
    endrow = middlerowvalues[0] + 300
        #middlerowvalues = (indices[length/2,:] + indices[length/2 + 1,:])/2 #als de lengte een even getal is, neem het gemiddelde van de twee middelste
    #middlerowvalues wordt nu gebruikt als middelpunt om een cirkelvormige mask over het lampje te zetten en daarmee de gemiddelde intensiteit te berekenen$
    r = 220 #Ruim geschat uit een van de geplotte afbeeldingen
    teller = 0
    n = 0
    new_image = image[int(startrow):int(endrow), int(startcol):int(endcol)]
    for k in range (int(startrow), int(endrow)):
        for l in range(int(startcol), int(endcol)):
            if ((l - middlerowvalues[1])**2 + (k - middlerowvalues[0])**2) <= r**2:
                teller += image[k,l]
                n += 1
    mean_intensity = teller / n
    return mean_intensity, new_image

def select_middle_row(image):
    length = len(image[:,0])
    if length % 2 != 0:
        middlerow = image[math.ceil(length),:]
    return middlerow

def select_every_nth(stringlist, beginvalue):
    lst = []
    for i in range(beginvalue, len(stringlist)+1, 10):
        if i < len(stringlist):
            #print(stringlist[i])
            lst.append(stringlist[i])
    return lst

def mean_intensity_filter(directory, filenames):
    int_array = np.zeros(len(filenames))
    i = 0
    for f in filenames:
        I, n = find_lamp_intensity(read_data(directory, f))
        int_array[i] = I
        if i == 2:
            x = np.array(n)
        i += 1
#    mean_intensity = np.mean(int_array)
    return int_array, x

filescirkel1 = filenames(path_calibr1)
filescirkel2 = filenames(path_calibr2)
filescirkel3 = filenames(path_calibr3)

#meandark = bad_pixels(path_calibr_darks)
"""Alle LHCP datapunten"""
everyLHCP_circle1 = select_every_nth(filescirkel1,4) #LHCP
everyLHCP_circle2 = select_every_nth(filescirkel2,4)
everyLHCP_circle3 = select_every_nth(filescirkel3,4)

circle1_LHCP_meanInt, x1 = mean_intensity_filter(path_calibr1, everyLHCP_circle1)
circle2_LHCP_meanInt, x2 = mean_intensity_filter(path_calibr2, everyLHCP_circle2)
circle3_LHCP_meanInt, x3 = mean_intensity_filter(path_calibr3, everyLHCP_circle3)

#intensity_LHCP = np.array([circle1_LHCP_meanInt, circle2_LHCP_meanInt, circle3_LHCP_meanInt])
angles = np.array([10, 15, 20])

lineA = np.array([circle1_LHCP_meanInt[0], circle2_LHCP_meanInt[0], circle3_LHCP_meanInt[0]])
lineB = np.array([circle1_LHCP_meanInt[1], circle2_LHCP_meanInt[1], circle3_LHCP_meanInt[1]])
lineC = np.array([circle1_LHCP_meanInt[2], circle2_LHCP_meanInt[2], circle3_LHCP_meanInt[2]])
lineD = np.array([circle1_LHCP_meanInt[7], circle2_LHCP_meanInt[3], circle3_LHCP_meanInt[5]])
lineE = np.array([circle1_LHCP_meanInt[6], circle2_LHCP_meanInt[4], np.nan])
lineF = np.array([circle1_LHCP_meanInt[5], circle2_LHCP_meanInt[5], circle3_LHCP_meanInt[3]])
lineG = np.array([circle1_LHCP_meanInt[4], circle2_LHCP_meanInt[6], np.nan])
lineH = np.array([circle1_LHCP_meanInt[3], circle2_LHCP_meanInt[7], circle3_LHCP_meanInt[4]])


plt.plot(angles, lineA, label = 'A')
plt.plot(angles, lineB, label = 'B')
plt.plot(angles, lineC, label = 'C')
plt.plot(angles, lineD, label = 'D')
plt.plot(angles, lineE, label = 'E')
plt.plot(angles, lineF, label = 'F')
plt.plot(angles, lineG, label = 'G')
plt.plot(angles, lineH, label = 'H')
plt.title('Intensity loss per position on different radii')
plt.xlabel('Radius of circle: 10cm - 15cm - 20cm')
plt.ylabel('Mean intensity of lamp')
plt.legend()
plt.show()

#plt.scatter(angles, intensity_LHCP)
#plt.plot(angles, intensity_LHCP)
#plt.xlabel('Radius of circle')
#plt.ylabel('Mean intensity value of the 8 positions')
#plt.title('Intensityloss of the camera for LHCP filter')
#plt.show()

"""Alle RHCP datapunten"""
def plot_intensity_radius(directory1, directory2, directory3, folder_cirkel1, folder_cirkel2, folder_cirkel3, result_file): 
    radius = np.array([10, 15, 20])
    filter_name = ['Q+', 'Q-', 'U+', 'U-', 'V+', 'V-']
    for j in range(0,6):
        files_circle1 = select_every_nth(folder_cirkel1,j) #volgorde: Qp, Qm, Up, Um, Vp, Vm
        files_circle2 = select_every_nth(folder_cirkel2,j)
        files_circle3 = select_every_nth(folder_cirkel3,j)
        circle1_meanInt = mean_intensity_filter(directory1, files_circle1)
        circle2_meanInt = mean_intensity_filter(directory2, files_circle2)
        circle3_meanInt = mean_intensity_filter(directory3, files_circle3)
        intensity_filter = np.array([circle1_meanInt, circle2_meanInt, circle3_meanInt])
        
        #plt.scatter(radius, intensity_filter)
        plt.plot(radius, intensity_filter)
        plt.xlabel('Radius of circle')
        plt.ylabel('Mean pixel_intensity value of the 8 positions')
        plt.title('Intensity-lossof the camera for'+filter_name[j]+'filter at R(cm)')
        plt.savefig(result_file + 'Int_loss_' + filter_name[j] + '.png')
    return

#plot_intensity_radius(path_calibr1, path_calibr2, path_calibr3, filescirkel1, filescirkel2, filescirkel3, result_lamp)
'''
everyRHCP_circle1 = select_every_nth(filescirkel1,5) #RHCP
everyRHCP_circle2 = select_every_nth(filescirkel2,5)
everyRHCP_circle3 = select_every_nth(filescirkel3,5)
angles = np.array([10, 15, 20])
circle1_RHCP_meanInt = mean_intensity_filter(path_calibr1, everyRHCP_circle1)
circle2_RHCP_meanInt = mean_intensity_filter(path_calibr2, everyRHCP_circle2)
circle3_RHCP_meanInt = mean_intensity_filter(path_calibr3, everyRHCP_circle3)

intensity_RHCP = np.array([circle1_RHCP_meanInt, circle2_RHCP_meanInt, circle3_RHCP_meanInt])

#plt.scatter(angles, intensity_RHCP)
plt.plot(angles, intensity_RHCP)
plt.xlabel('Radius of circle')
plt.ylabel('Mean intensity value of the 8 positions')
plt.title('Intensityloss of the camera for RHCP filter')
plt.show()'''

"""Alle Qp datapunten"""
"""
everyQm_circle1 = select_every_nth(filescirkel1,3) #RHCP
everyQm_circle2 = select_every_nth(filescirkel2,3)
everyQm_circle3 = select_every_nth(filescirkel3,3)
angles = np.array([10, 15, 20])
circle1_Qm_meanInt = mean_intensity_filter(path_calibr1, everyQm_circle1)
circle2_Qm_meanInt = mean_intensity_filter(path_calibr2, everyQm_circle2)
circle3_Qm_meanInt = mean_intensity_filter(path_calibr3, everyQm_circle3)

intensity_Qm = np.array([circle1_Qm_meanInt, circle2_Qm_meanInt, circle3_Qm_meanInt])

#plt.scatter(angles, intensity_Qp)
plt.plot(angles, intensity_Qm)
plt.xlabel('Radius of circle')
plt.ylabel('Mean intensity value of the 8 positions')
plt.title('Intensityloss of the camera for Um filter')
plt.show()"""