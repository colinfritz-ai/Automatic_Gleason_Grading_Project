import tensorflow as tf 
import tensorflow_io as tfio 
import matplotlib.pyplot as plt
import numpy as np 
import csv


def create_label_strings(filepath_to_train_csv, cloud = False):
	"""
	Description:
	This function creates a list of each of label as a string.  The label string at the 0th index 
	corresponds to the image_id and label stored in the first row of the train csv file 

	Args:
	filepath_to_train_csv = string filepath to the train csv file

	Returns:
	label_strings = list of label strings where the label string at the 0th index corresponds 
	to the image represented by the filepath at the 0th index of image_filepaths (the returned list of create_image_filepath_strings)
	"""

	label_strings = []
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
					label_strings.append(row[2])
					line_count+=1

	return label_strings


def create_image_filepath_strings(filepath_to_train_csv, cloud=False):
	"""
	Description:
	Takes filepath to the train csv file containing each image_id and it's corresponding isup grade 
	in columns 0 and 2 respectively 

	Args:
	filepath_to_train_csv = the filepath to the train csv file
	cloud = specifies whether to configure file path to look for image_id.tiff on local computer or google cloud storage bucket

	Returns:
	image_filepaths = list of file path strings to the tiff images 
	"""
	image_filepaths = []
	folder_path = '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/resized_images/' 
	bucket_path = ''
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
	return image_filepaths

def preprocess_images(img_path):
	"""
	Description:
	Reads tiff files from the given filepath and converts them to tensors consumable by the model

	Args:
	img_path = filepath (within our dataset element being processed) to the tiff image of interest 

	Returns: 
	img_decoded[:,:,0:3] = only the RGB channels of the RGBA img variable 
	"""
	img=tf.io.read_file(img_path)
	img_decoded=tfio.experimental.image.decode_tiff(img)
	return img_decoded[:,:,0:3]


def preprocess_label_strings(label_string):
	"""
	Description:
	Maps the input string to a one_hot encoded vector as a tensorflow tensor

	Args:
	label_string = string representing isup grade

	Returns:
	new = one_hot encoded tensorflow tensor e.g  if label_string == '1' new is [1,0,0,0,0]
	"""
	isup_labels = ['1', '2', '3', '4', '5']
	new_list = [x==label_string for x in isup_labels]
	ind=tf.argmax(new_list)
	new=tf.one_hot(ind,5)
	return new
	
def mapping_process(img,label):
	"""
	Description:
	Calls preprocess_images and preprocess_label_strings to 
	create the new image and label values for dataset elements

	Args:
	img = filepath to tiff image produced by create_image_filepath_strings
	label = string value representing the isup grade for image at img

	Returns:
	tissue_image = tensor RGB image representation to be stored in dataset element
	isup_grade =  one_hot encoded tensor to be stored in dataset element
	"""
	tissue_image=preprocess_images(img)
	isup_grade=preprocess_label_strings(label)
	return tissue_image,isup_grade

def configure_for_performance(ds):
	""" 
	Description:
	Configures the tf.Data.Dataset to be speed training

	Args:
	ds = tf.data dataset to be configured

	Returns:
	ds = reconfigured tf.data dataset
	"""
	ds = ds.cache()
	ds = ds.prefetch(2)
	return ds

def CNN_Model(input_shape):
	"""
	Description:
	Keras model for training on prostate tissue images 

	Args:
	input_shape = tuple specifying input dimensions of the model 

	Returns:
	model = Keras model instance to be compiled, fit, and predicted with 
	"""
	X_input = tf.keras.Input(input_shape)
	X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
	X = tf.keras.layers.Conv2D(32, (7, 7), strides = (1, 1), activation = 'relu', name = 'conv0')(X)
	X = tf.keras.layers.MaxPooling2D((20, 20), name='max_pool')(X)
	X = tf.keras.layers.Activation('relu')(X)
	X = tf.keras.layers.Flatten()(X)
	X = tf.keras.layers.Dense(5, activation='sigmoid', name='fc')(X)
	model = tf.keras.Model(inputs = X_input, outputs = X, name='HappyModel')
	return model

#Create lists of filepaths to image files and label values
train_csv_path = '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/train_test_csv_files/train.csv'
label_strings = create_label_strings(train_csv_path)
image_filepaths = create_image_filepath_strings(train_csv_path)

#parsing the lists into train and validation sets
train_label_strings = label_strings[0:int(len(label_strings)*0.8)]
train_image_filepaths = image_filepaths[0:int(len(label_strings)*0.8)]
validation_label_strings = label_strings[int(len(label_strings)*0.8):]
validation_image_file_paths = image_filepaths[int(len(label_strings)*0.8):]

#creating validation dataset with (image, label) elements as numpy arrays from the filepaths and strings in the original lists 
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_image_file_paths, validation_label_strings))
shuffled_validation_dataset = validation_dataset.shuffle(len(validation_label_strings))
mapped_validation_dataset = shuffled_validation_dataset.map(mapping_process)
repeating_validation_dataset = mapped_validation_dataset.repeat()
batched_validation_dataset = repeating_validation_dataset.batch(2)

#creating train dataset with (image, label) elements as numpy arrays from the filepaths and strings in the original lists
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_filepaths, train_label_strings))
shuffled_train_dataset = train_dataset.shuffle(len(train_label_strings)) 
mapped_train_dataset = shuffled_train_dataset.map(mapping_process)
repeating_train_dataset = mapped_train_dataset.repeat()
batched_train_dataset = repeating_train_dataset.batch(3)

#Creates model instance, compiles it, and then fits
ISUP_GRADING_MODEL=CNN_Model((512,512,3))
ISUP_GRADING_MODEL.compile(loss = 'categorical_crossentropy' , optimizer = 'SGD' , metrics =["accuracy"])
ISUP_GRADING_MODEL.fit(x=batched_train_dataset, epochs = 100 , verbose=1, validation_data = batched_validation_dataset, steps_per_epoch = 1, validation_steps = 1)











