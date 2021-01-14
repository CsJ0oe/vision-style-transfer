#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "[Simon Audrix, Tarek Lammouchi, Quentin Lanneau, Gabriel Nativel-Fontaine]"
__date__ = "20-12-17"
__usage__ = "Some useful functions"
__version__ = "1.0"

import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from skimage.transform import resize
import tensorflow as tf
from tensorflow.python.keras.applications import vgg19
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from numpy import savez_compressed


def buildNoiseImage(width, height, channels):
    """ Generate a random image with gaussian noise

    :param width:
    :param height:
    :param channels:
    :return:
    """
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (height, width, channels))
    gauss = gauss.reshape((1, height, width, channels))
    return tf.cast(gauss, tf.float32)


def preprocess_image(img, height, width):
    """ Open, resize and format a picture into appropriate tensors

    :param image_path:
    :param height:
    :param width:
    :return:
    """
    img = resize(img, (height, width))

    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(x, height, width):
    """ Convert a tensor into a valid image

    :param x:
    :param height:
    :param width:
    :return:
    """
    x = x.reshape((height, width, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x


# load all images in a directory into memory
def load_images_and_normalize(path, size=(256, 256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = (img_to_array(pixels) - 127.5) / 127.5
        # store
        data_list.append(pixels)
    return asarray(data_list)

