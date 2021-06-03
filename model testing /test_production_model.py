import tensorflow as tf
import os
from Resize_and_Save import prepare_TFRecords
import numpy as np
# gcp_bucket= "panda_dataset/"
# save_path = "gs://" + gcp_bucket + "train_production"
save_path = "/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/train_production"
model = tf.keras.models.load_model(save_path)
preparer = prepare_TFRecords()
# filenames = "gs://panda_dataset/test_tfrecords/cloud_test_tfrecord"


filenames = "/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/Resized_Datasets/train"
filenames = tf.io.gfile.glob(filenames+"*")
test = tf.data.TFRecordDataset(filenames)
# test = test.with_options(option_no_order)
# def isolate_images(image, label):
# 	return image
test = test.map(preparer.read_TFRecord)
test = test.batch(1)
# for image,label in test:
# 	print("label")
# 	print(label.numpy())
# test = test.map(isolate_images).batch(1)

# for image in test:
# 	print(image.numpy().shape)

# "/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/test_tfrecords/test_tfrecord"
predictions=model.evaluate(x=test, return_dict = True)
print(predictions)



