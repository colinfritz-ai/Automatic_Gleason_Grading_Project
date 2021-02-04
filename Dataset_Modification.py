import numpy as np 
import tifffile as tiff
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import csv
import tensorflow as tf
import tensorflow_io as tfio 

def create_image_filepath_and_id_strings(filepath_to_train_csv, image_folder='/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/images/'):
	"""
	Description:
	Takes filepath to the train csv file containing each image_id and it's corresponding isup grade 
	in columns 0 and 2 respectively.  Converts each image id to a string in image_id list.  Also generates 
	a list of string filepaths to each image 

	Args:
	filepath_to_train_csv = the filepath to the train csv file
	cloud = specifies whether to configure file path to look for image_id.tiff on local computer or google cloud storage bucket

	Returns:
	image_filepaths = list of file path strings to the tiff images
	image_ids = list of strings with each string being the image id corresponding 
	to the filepath at the same index in the image_filepaths list 
	"""

	image_filepaths = []
	image_ids = []
	with open(filepath_to_train_csv) as train_csv:
		train_csv = csv.reader(train_csv, delimiter=',')
		line_count = 0
		for row in train_csv:
			if line_count == 0:
				line_count+=1
			else:
				image_filepaths.append(folder_path + row[0] + '.tiff')
				line_count+=1
				image_ids.append(row[0])
	return image_filepaths, image_ids



def write_resized_images(image_filepaths, image_ids,resized_location='/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/resized_images/'):
	"""
	Description: 
	Takes two lists, one of image_filepaths, one of image_ids.  Resizes each image at each image filepath
	and writes the new image to a new location specified by resized_location

	Args:
	image_filepaths: list of image filepath strings
	image_ids: list of image id strings 
	resized_location: single string representing new folder to write each resized image 

	Returns:
	None
	"""
	path = 0
	for i in image_filepaths:
		tissue_sample = tiff.imread(image_filepaths[path])
		tissue_sample_resized=cv2.resize(tissue_sample, (512,512), interpolation = cv2.INTER_NEAREST)
		tiff.imwrite( resized_location + image_ids[path] + '.tiff', tissue_sample_resized, photometric='rgb')
		path+=1

#creating lists of filepaths to images and image_ids 
image_filepaths, image_ids=create_image_filepath_and_id_strings('/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/train_test_csv_files/train.csv')
write_resized_images(image_filepaths, image_ids)
