import tensorflow as tf
import os
gcp_bucket= "panda_dataset/"
save_path = "gs://" + gcp_bucket + "mnist_example"
model=tf.keras.models.load_model(save_path)

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28 * 28))
x_train = x_train.astype('float32') / 255

predictions=model.predict(x_train, steps = 1)

prediction = tf.math.argmax(predictions, axis=1)
equality = tf.math.equal(prediction, y_train)
accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
print(accuracy.numpy())