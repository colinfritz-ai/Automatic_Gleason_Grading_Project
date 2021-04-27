"""
Author:  Colin Fritz 
Date:  04/27/2021
Description:  Resize original TIFF images to smaller JPEG images.  Save them as TFRecord shards to 
enhance data ingestion speeds during training on Google AI Platform.  
"""
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

class prepare_TFRecords():

	def __init__(self):
		pass 

	def bytestring_feature(self,list_of_bytestrings):
  		return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

	def int_feature(self,list_of_ints): # int64
  		return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

	def float_feature(self,list_of_floats): # float32
  		return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

	def resize_and_encode(self,file_path, new_shape=[512,512]):
		img=tf.io.read_file(file_path)
		img_decoded=tfio.experimental.image.decode_tiff(img)
		rgb_img=img_decoded[:,:,0:3]
		rgb_img=tf.image.resize(rgb_img, size = new_shape)
		encoded_jpeg=tf.io.encode_jpeg(tf.cast(rgb_img, dtype=tf.uint8))
		encoded_jpeg = tf.io.serialize_tensor(encoded_jpeg).numpy()
		return encoded_jpeg

	def ohe_label(self,label_string):
		isup_labels = ['1', '2', '3', '4', '5']
		new_list = [x==label_string for x in isup_labels]
		ind=tf.argmax(new_list)
		new=tf.one_hot(ind,5, on_value = 1.0, off_value = 0.0)
		return new.numpy()

	def to_TFRecord(self, tfrecord_writer, encoded_image, label):
		feature = {
		"image": self.bytestring_feature([encoded_image]),     
		"label": self.float_feature(label.tolist())
		}
		return tf.train.Example(features=tf.train.Features(feature=feature))
	
	def read_TFRecord(observation):
		return


preparer=prepare_TFRecords()
shards = 1
shard_size = 1
filename = "/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/test_tfrecords/test_tfrecord"
for shard in range(shards):
	with tf.io.TFRecordWriter(filename) as out_file:
	    for i in range(shard_size):
		    image = preparer.resize_and_encode("/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/images/004391d48d58b18156f811087cd38abf.tiff")
		    label = preparer.ohe_label("2")
		    example = preparer.to_TFRecord(out_file,
		                            image, #already encoded as a byte_string
		                            label
		                          )
		    out_file.write(example.SerializeToString())



