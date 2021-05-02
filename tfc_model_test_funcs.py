"""
Authors:
Colin Fritz

Description:
Model definition, tf.data.Dataset pipeline setup, and training code.
"""
import csv
import tensorflow as tf
import tensorflow_io as tfio



def create_label_strings(filepath_to_train_csv, cloud = False):
    """
    Description:
    This function creates a list of each of label as a string.  The label string at the 0th index
    corresponds to the image_id and label stored in the first row of the train csv file

    Args:
    filepath_to_train_csv = string filepath to the train csv file

    Returns:
    label_strings = list of label strings where the label string at the 0th index corresponds
    to the image represented by the filepath at the 0th index of image_filepaths
    (the returned list of create_image_filepath_strings)
    """

    label_strings = []
    if not cloud:
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
    cloud = specifies whether to configure file path to look for image_id.tiff on
    local computer or google cloud storage bucket

    Returns:
    image_filepaths = list of file path strings to the tiff images
    """
    image_filepaths = []
    repo_path = '/Users/colinfritz/Desktop/my_repos/Automatic_Gleason_Grading_Project'
    folder_path = repo_path + '/resized_images/'
    if not cloud:
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
    new=tf.one_hot(ind,5, on_value = 1.0, off_value = 0.0)
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

def configure_for_performance(dataset):
    """
    Description:
    Configures the tf.Data.Dataset to be speed training

    Args:
    ds = tf.data dataset to be configured

    Returns:
    ds = reconfigured tf.data dataset
    """
    dataset = dataset.cache()
    dataset= dataset.prefetch(2)
    return dataset

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
    output = tf.keras.layers.MaxPooling2D((20, 20), name='max_pool')(output)
    output = tf.keras.layers.Activation('relu')(output)
    output = tf.keras.layers.Flatten()(output)
    output = tf.keras.layers.Dense(5, activation='sigmoid', name='fc')(output)
    model = tf.keras.Model(inputs = x_input, outputs = output, name='HappyModel')
    return model