# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 10:48:44 2019

#Flow plan

# Open folder and create a list of .NEF files
# Check if groups of 10 make sense (exif timing) and assign random name in list
# Before opening, check orientation of the image!
# For each set, calculate the polarization images
# Get ROI (drawer)
#Try to read label (not important now, just to show it can be done)
#Check polarization images for significant circular polarization inside ROI, and write result to file list ('histogram', as # pix per intensity bracket?)
# Segment circ polarization image
# process image without filter (7th) to auto WB, and label objects with significant circular polarization (bbox)
#Write result image to separate folder
#Write file list with results in readable format to same folder
# TIME PERMITTNG:
# Calculate rg chromaticity color space image, and compare between iridescence images

#End result: 4 images, (1) the image itself, (2) stretched (to a max) and heatmapped circ polariz, 
#(3) stretched and heatmapped iridescence, (4) AoLP 
 -file list with mean values, histograms

@author: arie
"""
    #%% Import libraries
    
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


#%% Open raw file in correct orientation
    
def openRawRotate(filePath):
    filePath = str(filePath)
    with imread(filePath) as rawimg:
        img = rawimg.postprocess()   #Get RGB image from raw using default settings
    #Rotate image to correct orientation
        import exifread
    f = open(filePath, 'rb')  # Open image file for reading (binary mode)
    tags = exifread.process_file(f, details=False)  #Return Exif tags. Not processing thumbnail and user tags to speed up processinig
    if "Image Orientation" in tags.keys():
        if str(tags['Image Orientation']) == 'Rotated 90 CW':
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if str(tags['Image Orientation']) == 'Rotated 90 CCW':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img  

#%% Define drawer finding function

def getroi(FilePath):
    #Get the roi of the drawer

    img = openRawRotate(FilePath)  #Open raw image
            
    gray = img[:,:,2]  #Take the blue layer of the image, to make sure the wood edge of the box (reddish) is dark
    blurred = cv2.GaussianBlur(gray,(21,21),0)      #Blur image a little to remove noise
    ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  #Threshold grayscale image
    threshinv = cv2.bitwise_not(thresh)  #First invert to make background an object
    kernel = np.ones((11,11),np.uint8) #define kernel for dilatation and opening
    threshinv = cv2.morphologyEx(threshinv, cv2.MORPH_OPEN, kernel)  #Open image to remove noise
    threshinv = cv2.dilate(threshinv,kernel,iterations = 7)   #Dilate to fill small gaps
    threshinv = cv2.erode(threshinv,kernel,iterations = 7)   #Dilate to fill small gaps
    #  Find the drawer by finding largest white blob (the background)
    connectivity = 4   #Use 4 or 8 connectivity
    output = cv2.connectedComponentsWithStats(threshinv, connectivity, cv2.CV_32S)  #Get connected components
    stats = output[2]        # The third cell is the stat matrix
    stats = stats[np.argsort(-stats[:,4])]  #sort by area (col 4) in descending order
 
    ##Get roi corresponding to drawer
    indices = []   #get a list of indices of x values in the right range
    for index, value in enumerate(stats[:,1]):
        if 0 <= value <= 1300                                            \
           and stats[index,3]  < 5999                                    \
           and 1.2<= (stats[index,2]/stats[index,3]) <= 1.34             \
           and 9000000 <= (stats[index,3]*stats[index,2]) <= 18000000:   \
            indices.append(index)
    indices.sort()  #Get first value, as that should corespond to the largest area (as stats is sorted in that order)
    try:
        roi = [stats[indices[0],0],stats[indices[0],1],stats[indices[0],2],stats[indices[0],3]]  #Save bounding box as ROI [x,y,w,h]
    except:
        print('\n','WARNING: Cannot find drawer- using predefined roi') 
        roi = [300, 400, 4610, 3600]   #Hard-coded roi for drawer TEMPORARY FIX
    return roi

#%%  Define label reading function

def getlabel(roi,FilePath):   
    from pylibdmtx.pylibdmtx import decode   #For reading label code
    img = openRawRotate(FilePath)  #Open raw image
    gray = img[:,:,2]  #Take the blue layer of the image, to make sure the wood edge of the box (reddish) is dark
    blurred = cv2.GaussianBlur(gray,(5,5),0)      #Blur image a little to remove noise
    masked = np.zeros(blurred.shape,np.uint8)  #Make a black image
    masked[roi[0]:roi[1],roi[3]:blurred.shape[1]] = blurred[roi[0]:roi[1],roi[3]:blurred.shape[1]]   #mask[y:y+h,x:x+w] = img[y:y+h,x:x+w]
    ret, thresh = cv2.threshold(masked,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)  #Threshold grayscale image
    threshinv = cv2.bitwise_not(thresh)  #First invert to make background an object
    kernel = np.ones((3,3),np.uint8)   #Erode foreground a bit
    threshinv = cv2.erode(threshinv,kernel,iterations = 2) 
    connectivity = 8   #Use 4 or 8 connectivity
    output = cv2.connectedComponentsWithStats(threshinv, connectivity, cv2.CV_32S)  #Get connected components

    stats = output[2]        # The third cell is the stat matrix
    stats = stats[np.argsort(-stats[:,4])]  #sort by area (col 4) in descending order
    ###Get labels and read drawer code
    labelrois = [0]   #get a list of indices with the right properties 
    #(to the right of drawer, label is about 1/11th of drawer, area greater than 40000px,
    # aspect ratio of about 3.4 )
    for index, value in enumerate(stats[:,0]):
        if roi[2] >= value \
           and roi[0] <= stats[index,1] <= roi[1]+roi[2] \
           and (stats[index,3]*stats[index,2]) >= 20000 \
           and 3 <= (stats[index,2]/stats[index,3]) <= 7:
           labelrois.append(index)
    del labelrois[0]  #Remove leading zero
    labelrois.sort(reverse=True)  #reverse list so search for code starts with smallest image to save time
    
    #stats[-1,4]  #area of largest connected object in pix
    #stats[-1,0]  #leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction
    #stats[-1,1]  #topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction
    #stats[-1,2]  #horizontal size of the bounding box
    #stats[-1,3]  #vertical size of the bounding box
    
    #Pass all labels to reader until DataMatrix is read (TO DO: if not, pass to tesseract for ocr)
    #print('Looking for drawer code..')
    Drawerlabel = ''
    for value in labelrois:
        subimg = gray[stats[value,1]:(stats[value,1]+stats[value,3]), stats[value,0]:(stats[value,0]+stats[value,2])]  #subimg = gray[y1:y2, x1:x2]
        subimg = cv2.rotate(subimg, cv2.ROTATE_90_COUNTERCLOCKWISE)  #Rotate image counterclockwise
        subimg = cv2.flip(subimg, 1)  #Mirror image to be readable
        ret, DMthresh = cv2.threshold(subimg,0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)  #Threshold grayscale image
        DMthresh = cv2.bitwise_not(DMthresh) 
        out = decode(DMthresh)
        if len(out) < 1:
            continue
        else:
            Drawerlabel = out[0][0][0:9].decode('UTF-8')  #Decode byte to string
            break
    if Drawerlabel == '':
        print('\n','Could not find drawer datamatrix, passing to OCR')
        #Pass to tesseract
        import pytesseract
        import re
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'   #path of tesseract executable
        for value in labelrois:
            subimg = gray[stats[value,1]:(stats[value,1]+stats[value,3]), stats[value,0]:(stats[value,0]+stats[value,2])]  #subimg = gray[y1:y2, x1:x2]
            subimg = cv2.rotate(subimg, cv2.ROTATE_90_COUNTERCLOCKWISE)  #Rotate image counterclockwise
            subimg = cv2.flip(subimg, 1)  #Mirror image to be readable
            #ret, DMthresh = cv2.threshold(subimg,0,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C+cv2.THRESH_OTSU)  #Threshold grayscale image
            #DMthresh = cv2.bitwise_not(DMthresh) 
            out = pytesseract.image_to_string(subimg)
            searchObj = re.search('[a-z]+[\.><\\\/][0-9]+', out, re.M|re.I)  #Search for string that looks like drawer code
            try: out = searchObj.group()  #Get string out
            except: out = ''
            if len(out) < 2:
                continue
            else:
                Drawerlabel = out
                print('\n', 'Tesserarct found label ', out)
                break
        if Drawerlabel == '':
            Drawerlabel = 'NoLabelFound'  #If something went wrong, give a default drawerlabel
        
    return Drawerlabel

#%% Read EXIF time label function

def exiftime(path_name): 
   from datetime import datetime
   import exifread
   # Open image file for reading (binary mode)
   f = open(path_name, 'rb')
   # Return Exif tags
   tags = exifread.process_file(f, details=False)  #Not processing thumbnail and user tags to speed up processinig
   if "Image DateTimeOriginal" in tags.keys():
       datetime_object = datetime.strptime(str(tags['Image DateTimeOriginal']), '%Y:%m:%d %H:%M:%S') #Parse date and time
   return datetime_object

#%% Generate random string function
    
def GenRandString(lenght):
    import string
    import random
    allchar = string.ascii_letters + string.digits
    randString = str( "".join(random.choice(allchar) for x in range(0,lenght) ))
    return randString
   
#%% Read file function

def read_data(filename):
    raw_data = imread(str(filename))
    raw_img = raw_data.raw_image.astype(np.float64)
    raw_img -= 150
    return raw_img    



#%% Use IO- Ask user to select image file NOTE dialog sometimes opens behind spyder window!
   
root = tk.Tk()
root.lift()
directory =  Path(filedialog.askdirectory(initialdir = "/",title = "Select Date Folder with subfolders with .NEF files"))
root.destroy()

#Get file locations in all subfolders
folders = next(os.walk(directory))[1]

for subfolder in folders:
    #t = time.time()
    CurSubFolder = Path(os.path.join(directory, subfolder))
    print('\n','Currently analyzing ' + str(CurSubFolder))
    fileSets = sorted(CurSubFolder.glob("*.NEF"))  #Get file names sorted by name
    print('\n', str(CurSubFolder) + ' contains ' + str(len(fileSets)) + ' NEF files')
    print('\n', 'This subfolder should take approximately ' + str(round(len(fileSets)/2.6,0)) + ' minutes')
    folder = CurSubFolder

    #%% Ask user to select image file NOTE dialog sometimes opens behind spyder window!
        
    #root = tk.Tk()
    #root.lift()
    #folder =  Path(filedialog.askdirectory(initialdir = "/",title = "Select Folder"))
    #root.destroy()
    
    #fileSets = sorted(folder.glob("*.NEF"))
    if len(fileSets) > 0:
        ResultsFolderPath = os.path.join(folder, 'Results')
        try: os.mkdir(ResultsFolderPath)  #Create folder to put result images
        except: # catch *all* exceptions
            print('Results folder already exists')
    
    #%% Run through file list and write time differences
    
    TimeDiff = [0]*len(fileSets)
    for index, path in enumerate(fileSets):
        if index == len(fileSets)-1:
            TimeDiff[index] = float(0)
            break
        diff = exiftime(fileSets[index+1]) - exiftime(fileSets[index])
        TimeDiff[index] = diff.total_seconds()  #difference in time in seconds
    print('\n', 'Time index created')
    fileSets = [fileSets,TimeDiff]
    
    #%% Label sets with random string
    
    setNames = ['NoLabel']*len(fileSets[0])   #Create empty list
    setName = GenRandString(10)  #Generate a random set name for the first set
    setCount = 0  #Number of the current set
    unique = [[setCount],[setName],[0]] #Create a list of sets to loop through later
    for index, path in enumerate(fileSets[0]):
        # Loop through, changing set name whenever a time difference > 10s
        if abs(fileSets[1][index]) <= 10:
            setNames[index] = setName
        if abs(fileSets[1][index]) > 10:
            setNames[index] = setName
            setName = GenRandString(10)  #Generate a random set name for the next set
            setNames[index+1] = setName
            setCount = setCount + 1 
            unique[0].append(setCount)
            unique[1].append(setName)
            unique[2].append(index+1)
    fileSets.append(setNames)  #Add label data to files list
    
    #%% Run through all sets
    
    unique.append([0]*len(unique[0])) #Add columns to write presence of pos and neg circpol, iridescence
    unique.append([0]*len(unique[0]))
    unique.append([0]*len(unique[0]))
    for setIndex, startIndex in enumerate(unique[2]):
        files = fileSets[0][startIndex:startIndex+10]
        t = time.time() #Track how long it takes to do one set
        print('Now analyzing set ' + str(setIndex+1) + ' of ' + str(len(unique[0])))
        
        #%% Get ROI of drawer
        
        roi = getroi(str(files[6])) #Gives (x,y,w,h)
        
        #%% Try to read file name and if succesful, update all files
        
        #drawerLabel = getlabel(roi,str(files[6]))  
        
        # %% Calculate polarization images (Olivier's functions)
        
        # Image arithmatic
        # I0-IR: intensity at 0/90/45/-45/left/right degrees
        # I_observed: no filter, same lighting as I0-IR
        # Il0, Il1, Il2: intensity without filter, at different angles
        Qp, Qm, Up, Um, Vp, Vm, I_observed, Il0, Il1, Il2 = [read_data(f) for f in files]
        colours = imread(str(files[0])).raw_pattern
        
        Q = Qp - Qm
        U = Up - Um
        V = Vp - Vm
        
        I_Q = Qp + Qm
        I_U = Up + Um
        I_V = Vp + Vm
        
        q = Q / I_Q
        u = U / I_U
        v = V / I_V
        
        #I_RGBG,_ = raw.pull_apart(I_observed, colours)
        q_RGBG,_ = raw.pull_apart(q, colours)
        u_RGBG,_ = raw.pull_apart(u, colours)
        v_RGBG,_ = raw.pull_apart(v, colours)
        DoLP_RGBG = np.sqrt(q_RGBG**2 + u_RGBG**2)
        AoLP_RGBG = np.rad2deg(0.5 * np.arctan2(u_RGBG, q_RGBG))
        DoP_RGBG = np.sqrt(q_RGBG**2 + u_RGBG**2 + v_RGBG**2)
        
        #%% Separate left- and righthander polarization (v) into two grayscale images
              
        v_RGBG_pos = v_RGBG*(v_RGBG>=0)   #Only positive values
        v_RGBG_neg = v_RGBG*(v_RGBG<=0)   #Only negative values  
        
        #Reshape images to uint8 and greyscale
        v_RGB_pos = np.zeros((2008,3008,3),np.uint8)  #Create empty array for image
        v_RGB_pos[:,:,0] = v_RGBG_pos[0,:,:]*255
        v_RGB_pos[:,:,1] = v_RGBG_pos[1,:,:]*255
        v_RGB_pos[:,:,2] = v_RGBG_pos[2,:,:]*255
        v_gray_pos = cv2.cvtColor(v_RGB_pos, cv2.COLOR_RGB2GRAY)   
        
        v_RGB_neg = np.zeros((2008,3008,3),np.uint8)  #Create empty array for image
        v_RGB_neg[:,:,0] = v_RGBG_neg[0,:,:]*-255
        v_RGB_neg[:,:,1] = v_RGBG_neg[1,:,:]*-255
        v_RGB_neg[:,:,2] = v_RGBG_neg[2,:,:]*-255
        v_gray_neg = cv2.cvtColor(v_RGB_neg, cv2.COLOR_RGB2GRAY)   
        
    #    #Scale to full color range. Since image is not yet cropped, equalization is limited by polarization calibration light
    #    v_gray_pos = cv2.equalizeHist(v_gray_pos) 
    #    v_gray_neg = cv2.equalizeHist(v_gray_neg)
           
        #Scale images up to correspond to origional
        v_gray_pos = cv2.resize(v_gray_pos, (6016,4016), interpolation = cv2.INTER_AREA) 
        v_gray_neg = cv2.resize(v_gray_neg, (6016,4016), interpolation = cv2.INTER_AREA) 
    
        #Crop images to ROI
        v_gray_pos = v_gray_pos[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[2]]
        v_gray_neg = v_gray_neg[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[2]]
        
        #Calculate pos and neg histograms for reporting area and intensities to file
        hist_pos = cv2.calcHist([v_gray_pos],[0],None,[10],[0,256])
        hist_neg = cv2.calcHist([v_gray_neg],[0],None,[10],[0,256])
        
        if sum(hist_pos[5:10,0]) > 1000 or sum(hist_pos[8:10,0]) > 150:
            unique[3][setIndex] =  sum(hist_pos[5:10,0])
            print('significant positive polarization found in set ' + unique[1][setIndex])
        if sum(hist_neg[5:10,0]) > 1000 or sum(hist_neg[8:10,0]) > 150:
            unique[4][setIndex] =  sum(hist_neg[5:10,0])
            print('significant negative polarization found in set ' + unique[1][setIndex])        
       
        #%% #Create image to show beetles
        
        img = openRawRotate(str(files[6]))    #Get the image without filter  
        
        #TO DO: (Auto) whitebalance the image    
    
        DrawImage = img[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[2]]    #Crop image to ROI
        
        #Reduce image size to save disk space
        #DrawImage = cv2.resize(DrawImage, (3008,2008), interpolation = cv2.INTER_AREA) 
        #_RGB = cv2.resize(v_RGB, (3008,2008), interpolation = cv2.INTER_AREA)
        
        
       #%% Create single RGB image with circ pol info
        
        v_RGB = np.zeros_like(DrawImage,np.uint8)  #Create empty array for image
        v_RGB[:,:,0] = np.zeros_like(v_gray_pos[:,:])
        v_RGB[:,:,1] = v_gray_pos[:,:]
        v_RGB[:,:,2] = v_gray_neg[:,:]
        
        
        #Join two images together horizontally
        topImage = np.concatenate((DrawImage, v_RGB), axis=1)
        
           ##%% Create single RGB image with circ pol info          
    #    
    #    v_RGB = np.zeros((2008,3008,3),np.uint8)  #Create empty array for image
    #    v_RGB[:,:,0] = (v_RGBG[0,:,:]+1)*127
    #    v_RGB[:,:,1] = (v_RGBG[1,:,:]+1)*127
    #    v_RGB[:,:,2] = (v_RGBG[2,:,:]+1)*127
    #    v_RGB = cv2.resize(v_RGB, (6016,4016), interpolation = cv2.INTER_AREA)  #Resize to full
    #    v_RGB = v_RGB[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[2]]    #Crop image to ROI
        
        #%% Calculate iridescence image
            
        img7 = openRawRotate(str(files[7]))
        img8 = openRawRotate(str(files[8]))
            
        #First open and convert first two iridescence images to lab colorspace
        lab_image7 = cv2.cvtColor(img7, cv2.COLOR_BGR2LAB)
        lab_image8 = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
          
        #Calculate distances per color dimension (Green to Magenta and Blue to Yellow)
        colDist1 = lab_image7[:,:,1].astype(np.int16) - lab_image8[:,:,1].astype(np.int16)
        colDist2 = lab_image7[:,:,2].astype(np.int16) - lab_image8[:,:,2].astype(np.int16)
    
        #then calculate the 2d distance in lab color space
        colorDist = (colDist1**2 + colDist2**2)**0.5  #Calculate Euclidian distance
        iridescenceImage = np.array((colorDist/colorDist.max()) * 255, dtype = np.uint8) #Convert back to grayscale image for display
        #Convert to RGB image with color map
        
        #Reduce image size to save disk space
        iridescenceImage = iridescenceImage[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[2]]    #Crop image to ROI
        iridescenceImage = cv2.applyColorMap(iridescenceImage, cv2.COLORMAP_JET)
       
        #%% Create RGB image of angle of lin polarization
        
        AoLP_RGB = np.zeros((2008,3008,3),np.uint8)  #Create empty array for image
        AoLP_RGB[:,:,0] = (AoLP_RGBG[0,:,:]+90)*(255/180)
        AoLP_RGB[:,:,1] = (AoLP_RGBG[1,:,:]+90)*(255/180)
        AoLP_RGB[:,:,2] = (AoLP_RGBG[2,:,:]+90)*(255/180)
        AoLP_RGB = cv2.resize(AoLP_RGB, (6016,4016), interpolation = cv2.INTER_AREA)  #Resize to full
        AoLP_RGB = AoLP_RGB[roi[0]:roi[0]+roi[3],roi[1]:roi[1]+roi[2]]    #Crop image to ROI
    
    
        #%% Write images to a folder, together with circpolarization images
    
        #Join two images together horizontally
        botImage = np.concatenate((iridescenceImage, AoLP_RGB), axis=1)    
        
        #Join two images together vertically
        finalImage = np.concatenate((topImage, botImage), axis=0)
        
        width = int(finalImage.shape[1] * (2400/finalImage.shape[1]))
        height = int(finalImage.shape[0] * (2400/finalImage.shape[1]))
        dim = (width, height) 
        finalImage = cv2.resize(finalImage, dim, interpolation = cv2.INTER_AREA)  #Resize to smaller file
        
        DestFile = os.path.join(ResultsFolderPath, os.path.basename(files[0]))
        cv2.imwrite(str(DestFile + '_out.png'), finalImage, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elapsed = time.time() - t
        print('This set took ' + str(round(elapsed,2)) + ' seconds')
        print('Now writing result to file ' + str(os.path.basename(files[0]) + '_out.png') )
        
        
        #%% Write text file
            
        # os.chdir(ResultsFolderPath)  #Move to base dir and write text file
        # file = open("FileList.txt", "w")
        # for i in range(0,len(files)):
        #     file.write(str(files[i]) + '\t' + str(int(files[1][i])) + '\t' + str(files[2][i]) + '\n')
        # file.close() #This close() is important 
        
        #%%#Show draw image for debugging
        
        # cv2.namedWindow("pic", cv2.WINDOW_NORMAL) 
        # cv2.imshow('pic',finalImage) 
        # cv2.waitKey() #Show image for 2 seconds
        # cv2.destroyAllWindows()
        
    
    print('All done with files in folder ' + str(folder) )
    
print('Finished analyzing all subfolders ' + str(folders) )
#Beep to know folder is done.
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1500  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

    
        #%%
        # Optional:
        # Convert lin polarized images to RG(b) Chromaticity
        # Calculate RG Chromaticity colorspace distance between lin polarization images
        # Calculate color space distances between pixels and save as grayscale image
        # segment grayscale image
        # Highlight linear polarizing objects with e.g. black bounding box 
        # Optional 2:
        # Calculate chromaticity diff between iridescence images
        # Highlight iridescent objects (bbox) in separate iridescence image
        # Add EXIF file labels to allow sorting (e.g. by presence of polarization/iridescence) in windows explorer
