"""
Experimenting with how to use PIL, Numpy, and Matplotlib to load Tiff image files in as Numpy arrays representing RGB images

Note:  Should be confident that loading the tiff file does not result in significant data loss 

"""

import tifffile as tiff
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
# openslide allows for viewing of different levels and crops of the images like under a microscope with a slide
#import openslide 
tissue_sample_mask_filename = '/Users/colinfritz/Desktop/0005f7aaab2800f6170c399693a96917_mask.tiff'
tissue_sample_filename = '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/trial_tissue.tiff'
tissue_sample = tiff.imread(tissue_sample_filename)
tissue_sample_mask = tiff.imread(tissue_sample_mask_filename)
#print("mask_type: " + str(tissue_sample_mask.dtype))
print("sample_type: " + str(tissue_sample.dtype))
#tissue_sample = cv2.resize(tissue_sample , dsize=(2000, 2000))
print("number_items: " + str(tissue_sample.size))
print("itemsize_bytes: " + str(tissue_sample.itemsize))
print("megabytes: " + str((tissue_sample.size*tissue_sample.itemsize)/1000000))
# The array is showing all RGB values as 255.  This is incorrect behavior.  
#print("image_as_array: " + str(tissue_sample))
# trying to print the image also does not work in matplotlib despite it being a valid numpy array.  
plt.figure()
plt.imshow(tissue_sample)
plt.show()
# importing tiffile & numpy packages






