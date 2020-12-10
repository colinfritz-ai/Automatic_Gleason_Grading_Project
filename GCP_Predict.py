import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import numpy as np
import matplotlib.pyplot as plt
import sys 
import tensorflow as tf

def predict_json(project, model, instances, version=None):

    endpoint = 'https://us-central1-ml.googleapis.com'
    client_options = ClientOptions(api_endpoint=endpoint)
    ml = googleapiclient.discovery.build('ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    request = ml.projects().predict(
    name=name,
    body={ 'instances': instances })

    response = request.execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response

CLOUD_PROJECT = 'tf2servetutorial'
MODEL = 'fashion_mnist_new_trial'

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
test_predictions = predict_json(CLOUD_PROJECT, MODEL, test_images[:200].tolist(),'v1')
second_break = 0
for out_vector in test_predictions['predictions']:
    if second_break == 1:
        break
    for num in out_vector:
        if 0.0 < num < 1.0:
            print("found_non_binary: " + str(num))
            second_break = 1 
            break

#print(test_predictions)

plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


