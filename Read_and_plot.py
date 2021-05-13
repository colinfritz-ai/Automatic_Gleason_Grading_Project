from Resize_and_Save import prepare_TFRecords
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

option_no_order = tf.data.Options()
option_no_order.experimental_deterministic = False
preparer=prepare_TFRecords()
filename = "/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/test_tfrecords/test_tfrecord"
dataset4 = tf.data.TFRecordDataset(filename)
dataset4 = dataset4.with_options(option_no_order)
dataset4 = dataset4.map(preparer.read_TFRecord)
dataset4=dataset4.take(1)
for image, label in dataset4:
	print(image.numpy().shape)
	# plt.show()

