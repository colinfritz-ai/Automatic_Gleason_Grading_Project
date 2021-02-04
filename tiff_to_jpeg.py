import tifffile as tiff
import numpy as np 
import tensorflow_io as tfio 
import matplotlib.pyplot as plt
import cv2 
from PIL import Image 

tissue_sample_filename = '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/images/000920ad0b612851f8e01bcc880d9b3d.tiff'
tissue_sample = tiff.imread(tissue_sample_filename)
print(tissue_sample.shape)
img = Image.fromarray(tissue_sample)
img.save("geeks.jpg")





