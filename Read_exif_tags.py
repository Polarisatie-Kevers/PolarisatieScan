# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:22:14 2019

@author: arie
"""

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

