import tensorflow as tf
import tensorflow_cloud as tfc
from Resize_and_Save import prepare_TFRecords
import os

def cnn_model(input_shape):
    """
    Description:
    Keras model for training on prostate tissue images

    Args:
    input_shape = tuple specifying input dimensions of the model

    Returns:
    model = Keras model instance to be compiled, fit, and predicted with
    """
    x_input = tf.keras.Input(input_shape)
    output = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)
    output = tf.keras.layers.Conv2D(32,(7,7),strides=(1, 1),activation='relu',name='conv0')(output)
    output = tf.keras.layers.Conv2D(32,(7,7),strides=(1, 1),activation='relu',name='conv1')(output)
    output = tf.keras.layers.MaxPooling2D((20, 20), name='max_pool')(output)
    output = tf.keras.layers.Activation('relu')(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(100, activation='relu', name='fc_1')(output)
    output = tf.keras.layers.Dense(5, activation='sigmoid', name='fc_2')(output)
    model = tf.keras.Model(inputs = x_input, outputs = output, name='HappyModel')
    return model

preparer = prepare_TFRecords()
# filenames = "gs://panda_dataset/test_tfrecords/cloud_test_tfrecord"
filenames = "/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/Resized_Datasets/train_"
filenames = tf.io.gfile.glob(filenames+"*")
train_set = tf.data.TFRecordDataset(filenames)
train_set = train_set.repeat()
train_set = train_set.map(preparer.read_TFRecord, num_parallel_calls=tf.data.AUTOTUNE)
train_set = train_set.batch(200)


filenames = "/Volumes/external_1tb/Automated_Gleason_Grading_Project_Resources/Resized_Datasets/validation_"
filenames = tf.io.gfile.glob(filenames+"*")
validation_set = tf.data.TFRecordDataset(filenames)
validation_set = validation_set.map(preparer.read_TFRecord, num_parallel_calls = tf.data.AUTOTUNE)
validation_set = validation_set.batch(200)


model = cnn_model((512,512,3))

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            name='categorical_crossentropy'
            ),
            
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy'])


if tfc.remote():
    model.fit(x=train_set, validation_data=validation_set, callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=50, verbose=0,
            mode='auto', baseline=None, restore_best_weights=False
            )], steps_per_epoch = 100, epochs=2)
    gcp_bucket= "panda_dataset/"
    save_path = os.path.join("gs://", gcp_bucket, "train_production")
    model.save(save_path)

else:
    model.fit(x=train_set, validation_data=validation_set, callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=50, verbose=2,
            mode='auto', baseline=None, restore_best_weights=False
            )], steps_per_epoch = 100, epochs=2)
    model.save("/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project/train_production")
