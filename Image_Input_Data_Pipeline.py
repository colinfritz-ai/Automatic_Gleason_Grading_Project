"""
Experimenting with how to use PIL, Numpy, and Matplotlib to load Tiff image files in as Numpy arrays representing RGB images

Note:  Should be confident that loading the tiff file does not result in significant data loss 

"""

import tifffile as tiff
import numpy as np 
import tensorflow_io as tfio 
import matplotlib.pyplot as plt
import cv2 
 
tissue_sample_filename = '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/images/000920ad0b612851f8e01bcc880d9b3d.tiff'
tissue_sample = tiff.imread(tissue_sample_filename)

# The array is showing all RGB values as 255.  This is incorrect behavior.  
#print("image_as_array: " + str(tissue_sample))
# trying to print the image also does not work in matplotlib despite it being a valid numpy array.  
#plt.figure()
#plt.imshow(tissue_sample)
#plt.show()


# pts1 = np.float32([[5770,11800],[7820,13000],[15000,2500],[14000,1500]])
# pts2 = np.float32([[0,0],[3000,0],[0,12000],[3000,12000]])
# M = cv2.getPerspectiveTransform(pts1,pts2)
# dst = cv2.warpPerspective(tissue_sample[:,:,0],M,(3000,12000))
# dst2 = cv2.warpPerspective(tissue_sample[:,:,1],M,(3000,12000))
# dst3 = cv2.warpPerspective(tissue_sample[:,:,2],M,(3000,12000))
# final = np.stack((dst,dst2,dst3), axis =2)
# plt.subplot(121)
# plt.imshow(tissue_sample[:,:,0])
# plt.title('Input')
# plt.subplot(122)
# plt.imshow(tissue_sample[:,:,1])
# plt.title('Output')
# plt.show()

# plt.figure()
# plt.imshow(tissue_sample[:,:,0])
# plt.show()

# plt.figure()
# plt.imshow(tissue_sample[:,:,1])
# plt.show()

# plt.figure()
# plt.imshow(tissue_sample[:,:,2])
# plt.show()

shape = (tissue_sample[:,:,0].shape[1], tissue_sample[:,:,0].shape[0]) # cv2.warpAffine expects shape in (length, height)

matrix = cv2.getRotationMatrix2D(center=(11000,7900), angle=90, scale=1)
image1 = cv2.warpAffine( src=tissue_sample[:,:,0], M=matrix, dsize=shape )
image2 = cv2.warpAffine( src=tissue_sample[:,:,1], M=matrix, dsize=shape )
image3 = cv2.warpAffine( src=tissue_sample[:,:,2], M=matrix, dsize=shape )

final = np.stack((image1,image2,image3), axis =2)
plt.subplot(121)
plt.imshow(tissue_sample)
plt.title('Input')
plt.subplot(122)
plt.imshow(final)
plt.title('Output')
plt.show()


