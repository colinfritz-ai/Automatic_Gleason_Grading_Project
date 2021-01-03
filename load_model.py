import tensorflow as tf
import googleapiclient.discovery
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import sys 

# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# train_images = train_images / 255.0
# print(train_images[0].shape)
# test_images = test_images / 255.0
# new_model = tf.keras.models.load_model('fashion_mnist')
# predictions=new_model.predict(train_images[0:2])
# print("predictions: " + str(predictions))
node = [1,2,3]
a = node[-1]
node.append(4)
print(a)

