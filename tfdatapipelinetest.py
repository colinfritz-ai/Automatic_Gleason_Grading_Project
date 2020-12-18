import tensorflow as tf 
import tensorflow_io as tfio 
import matplotlib.pyplot as plt
import numpy as np 
import csv
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

def create_label_strings(filepath_to_train_csv, cloud = False):
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
	image_filepaths = []
	folder_path = '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/images/' 
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

def preprocess_images(img_path, img_height=10000, img_width=10000):
	img=tf.io.read_file(img_path)
	img_decoded=tfio.experimental.image.decode_tiff(img)
	#use slicing to make sure the tf.resize function works properly.  
	#think of how to reduce white space in image to lessen information loss per image 
	img_decoded=tf.image.resize(img_decoded[:,:,0:3],[img_height, img_width], method = 'nearest')
	return img_decoded


def preprocess_label_strings(label_string):
	isup_labels = ['1', '2', '3', '4', '5']
	one_hot = label_string == isup_labels
	return tf.argmax(one_hot)
	
def mapping_process(img,label):
	tissue_image=preprocess_images(img)
	isup_grade=preprocess_label_strings(label)
	return tissue_image,isup_grade

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def CNN_Model(input_shape):
	X_input = Input(input_shape)
	X = ZeroPadding2D((3, 3))(X_input)
	X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
	X = BatchNormalization(axis = 3, name = 'bn0')(X)
	X = MaxPooling2D((2, 2), name='max_pool')(X)
	X = Activation('relu')(X)
	X = Flatten()(X)
	X = Dense(1, activation='sigmoid', name='fc')(X)
	model = Model(inputs = X_input, outputs = X, name='HappyModel')

	return model

train_csv_path= '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/train_test_csv_files/train.csv'
label_strings= create_label_strings(train_csv_path)
image_filepaths= create_image_filepath_strings(train_csv_path)
train_label_strings = 	label_strings[0:int(len(label_strings)*0.8)]
train_image_filepaths = image_filepaths[0:int(len(label_strings)*0.8)]
validation_label_strings = label_strings[int(len(label_strings)*0.8):]
validation_image_file_paths = image_filepaths[int(len(label_strings)*0.8):]
original_dataset= tf.data.Dataset.from_tensor_slices((train_image_filepaths, train_label_strings))
original_dataset=original_dataset.shuffle(len(train_label_strings))
mapped_dataset = original_dataset.map(mapping_process)
repeating_dataset=mapped_dataset.repeat()
batched_dataset=repeating_dataset.batch(3)
# print_count = 0
# for element in mapped_dataset.as_numpy_iterator():
# 	if print_count == 0:
# 		print_count+=1
# 		continue
# 	else:
# 		print(element[0].shape)
# 		plt.figure()
# 		plt.imshow(element[0])
# 		plt.show()
# 		break









