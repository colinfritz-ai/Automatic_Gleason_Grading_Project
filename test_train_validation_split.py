"""
Randomly shuffling dataset to get train,validation, and test sets.
"""
import random
import csv 
from Resize_and_Save import prepare_TFRecords
import tensorflow as tf

images_and_labels=[]
with open("/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/prostate-cancer-grade-assessment/train.csv", "r") as f:
	rows=csv.reader(f, delimiter=',')
	line_count = 0
	for row in rows:
		if line_count>0:
			row_container = []
			row_container.append("/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/prostate-cancer-grade-assessment/train_images/"+ row[0].strip()+".tiff")
			row_container.append(row[2].strip())
			images_and_labels.append(row_container)
		else:
			line_count+=1


random.shuffle(images_and_labels)
total_num_images=len(images_and_labels)
num_train = int(total_num_images*.6)
num_validation = num_train + int(total_num_images*.2)
train = images_and_labels[:num_train]
validation = images_and_labels[num_train: num_validation]
test = images_and_labels[num_validation:]



#writing train set images
with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/train_images.txt', 'w') as filehandle:
    for listitem in train:
        filehandle.write('%s\n' % listitem[0])
#writing validation set images
with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/validation_images.txt', 'w') as filehandle:
    for listitem in validation:
        filehandle.write('%s\n' % listitem[0])
#writing test set images
with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/test_images.txt', 'w') as filehandle:
    for listitem in test:
        filehandle.write('%s\n' % listitem[0])


#writing train set labels
with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/train_labels.txt', 'w') as filehandle:
    for listitem in train:
        filehandle.write('%s\n' % listitem[1])
#writing validation set labels
with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/validation_labels.txt', 'w') as filehandle:
    for listitem in validation:
        filehandle.write('%s\n' % listitem[1])
#writing test set labels
with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/test_labels.txt', 'w') as filehandle:
    for listitem in test:
        filehandle.write('%s\n' % listitem[1])


if __name__ == "__main__":
	preparer = prepare_TFRecords()
	shard_size = 400
	# #train
	# filenames = []
	# with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/train_images.txt', "r") as f:
	# 	for i in range(len(train)):
	# 		filenames.append(f.readline().strip())

	# labels = []
	# with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/train_labels.txt', "r") as f:
	# 	for i in range(len(train)):
	# 		labels.append(f.readline().strip())

	# file_count = 0
	# for image_path, label_string in zip(filenames,labels):
	# 	if file_count%shard_size == 0 or file_count==0:
	# 		filename = "/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/Resized_Datasets/train_" + str(file_count)
	# 	with tf.io.TFRecordWriter(filename) as out_file:
	# 	    image = preparer.resize_and_encode(filenames[i])
	# 	    label = preparer.ohe_label(labels[i])
	# 	    example = preparer.to_TFRecord(out_file,
	# 	                            image, #already encoded as a byte_string
	# 	                            label
	# 	                          )
	# 	    out_file.write(example.SerializeToString())
	# 	file_count+=1


	#validation
	filenames = []
	with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/validation_images.txt', "r") as f:
		for i in range(len(validation)):
			filenames.append(f.readline().strip())

	labels = []
	with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/validation_labels.txt', "r") as f:
		for i in range(len(validation)):
			labels.append(f.readline().strip())


	file_count = 0
	for image_path, label_string in zip(filenames,labels):
		if file_count%shard_size == 0 or file_count==0:
			filename = "/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/Resized_Datasets/validation_" + str(file_count)
		with tf.io.TFRecordWriter(filename) as out_file:
		    image = preparer.resize_and_encode(filenames[i])
		    label = preparer.ohe_label(labels[i])
		    example = preparer.to_TFRecord(out_file,
		                            image, #already encoded as a byte_string
		                            label
		                          )
		    out_file.write(example.SerializeToString())
		file_count+=1

	#test
	filenames = []
	with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/test_images.txt', "r") as f:
		for i in range(len(test)):
			filenames.append(f.readline().strip())

	labels = []
	with open('/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/test_labels.txt', "r") as f:
		for i in range(len(test)):
			labels.append(f.readline().strip())

	
	file_count = 0
	for image_path, label_string in zip(filenames,labels):
		if file_count%shard_size == 0 or file_count==0:
			filename = "/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/Resized_Datasets/test_" + str(file_count)
		with tf.io.TFRecordWriter(filename) as out_file:
		    image = preparer.resize_and_encode(filenames[i])
		    label = preparer.ohe_label(labels[i])
		    example = preparer.to_TFRecord(out_file,
		                            image, #already encoded as a byte_string
		                            label
		                          )
		    out_file.write(example.SerializeToString())
		file_count+=1

	




