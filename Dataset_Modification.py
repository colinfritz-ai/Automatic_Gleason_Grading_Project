import numpy as np 
import tifffile as tiff
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import csv
import tensorflow as tf
import tensorflow_io as tfio 

def create_image_filepath_and_id_strings(filepath_to_train_csv, cloud=False):
	"""
	Description:
	Takes filepath to the train csv file containing each image_id and it's corresponding isup grade 
	in columns 0 and 2 respectively 

	Args:
	filepath_to_train_csv = the filepath to the train csv file
	cloud = specifies whether to configure file path to look for image_id.tiff on local computer or google cloud storage bucket

	Returns:
	image_filepaths = list of file path strings to the tiff images
	image_ids = list of strings with each string being the image id corresponding 
	to the filepath at the same index in the image_filepaths list 
	"""

	image_filepaths = []
	folder_path = '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/images/' 
	bucket_path = ''
	image_ids = []
	if cloud == False:
		with open(filepath_to_train_csv) as train_csv:
			train_csv = csv.reader(train_csv, delimiter=',')
			line_count = 0
			for row in train_csv:
				if line_count == 0:
					line_count+=1
				elif line_count>12:
					break
				else:
					image_filepaths.append(folder_path + row[0] + '.tiff')
					line_count+=1
					image_ids.append(row[0])
	return image_filepaths, image_ids


#creating lists of filepaths to images and image_ids 
image_filepaths, image_ids=create_image_filepath_and_id_strings('/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/train_test_csv_files/train.csv')

#loads each original tiff image then resizes it and writes it to new dataset directory
path = 0
for i in image_filepaths:
	tissue_sample = tiff.imread(image_filepaths[path])
	tissue_sample_resized=cv2.resize(tissue_sample, (512,512), interpolation = cv2.INTER_NEAREST)
	tiff.imwrite('/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/resized_images/' + image_ids[path] + '.tiff', tissue_sample_resized, photometric='rgb')
	#tissue_sample = tiff.imread('/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/resized_images/' + image_ids[path] + '.tiff')
	# img=tf.io.read_file('/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/resized_images/' + image_ids[path] + '.tiff')
	# img_decoded=tfio.experimental.image.decode_tiff(img)
	# print(img_decoded.shape)
	# plt.figure()
	# plt.imshow(img_decoded)
	# plt.show()
	path+=1


