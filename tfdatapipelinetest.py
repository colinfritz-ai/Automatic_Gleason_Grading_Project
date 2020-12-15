import tensorflow as tf 
import tensorflow_io as tfio 
import matplotlib.pyplot as plt
import numpy as np 

#file=tf.io.read_file('/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/images/trial_tissue.tiff')
#file=tfio.experimental.image.decode_tiff(file)
#print(file[:,:,4])
#plt.figure()
#plt.imshow(file)
#plt.show()
"""
this setup can use a helper function to run map on the dataset.  converting images to tensors with 
tensorflows io API and then aligning them with labels as one-hot encoded vectors 
"""
image_data=tf.data.Dataset.from_tensor_slices((['trial_tissue.tiff', 'second', 'third'],['3', '2', '4']))
for element in image_data.as_numpy_iterator():
	print(element)

