#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "[Simon Audrix, Tarek Lammouchi, Quentin Lanneau, Gabriel Nativel-Fontaine]"
__date__ = "20-12-17"
__usage__ = "Neural style transfer algorithm from Gaty"
__version__ = "1.0"

import time

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from IPython.display import Image, display

import numpy as np

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.applications import vgg19
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from src.utils import deprocess_image


def transfer_style(nb_epochs, content, style, style_layers, content_layers, name):
    """ Apply neural transfert style algorithm

    :param nb_epochs: How many epochs
    :param content: content image
    :param style: style image
    :param style_layer_names: layers to use for style loss
    :param content_layer_name: layers to use for content loss
    :param name: name for saving images
    :return:
    """

    def gram_matrix(x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    def style_loss(style, combination, height, width):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = height * width
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    def content_loss(base, combination):
        return tf.reduce_sum(tf.square(combination - base))

    def total_variation_loss(x, height, width):
        a = tf.square(x[:, : height - 1, : width - 1, :] - x[:, 1:, : width - 1, :])
        b = tf.square(x[:, : height - 1, : width - 1, :] - x[:, : height - 1, 1:, :])
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    # Using the convolutional base of VGG19, loaded with pre-trained ImageNet weights
    vgg = vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names)
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
    outputs = [layer.name for layer in vgg.layers]

    # Set up a model that returns the activation values for every layer in VGG19 (as a dict)
    feature_extractor = Model(inputs=vgg.inputs, outputs=outputs_dict)

    #vgg.summary()

    total_variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8

    def compute_loss(combination_image, base_image, style_reference_image, height, width):
        input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
        features = feature_extractor(input_tensor)

        loss = tf.zeros(shape=())

        # Content loss
        for l in content_layers:
            layer_features = features[outputs[l]]
            base_image_features = layer_features[0, :, :, :]
            combination_features = layer_features[2, :, :, :]
            loss += (content_weight / len(content_layers)) * content_loss(base_image_features, combination_features)

        # Style loss
        for l in style_layers:
            layer_features = features[outputs[l]]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features, height, width)
            loss += (style_weight / len(style_layers)) * sl

        loss += total_variation_weight * total_variation_loss(combination_image, height, width)
        return loss

    @tf.function
    def compute_loss_and_grads(combination_image, base_image, style_reference_image, height, width):
        with tf.GradientTape() as tape:
            loss = compute_loss(combination_image, base_image, style_reference_image, height, width)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    _, height, width, _ = content.shape
    gen_height = 400
    gen_width = int(width * gen_height / height)

    optimizer = SGD(ExponentialDecay(initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))
    combination_image = tf.Variable(content)

    # Training loop
    for epoch in range(1, nb_epochs + 1):
        loss, grads = compute_loss_and_grads(combination_image, content, style, gen_height, gen_width)
        optimizer.apply_gradients([(grads, combination_image)])
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{nb_epochs}], loss: {loss:.2f}")

    # Save final image
    final_img = deprocess_image(combination_image.numpy(), gen_height, gen_width)
    result_image_path = f"img/img_{name}_{nb_epochs}.png"
    save_img(result_image_path, final_img)

    display(Image(result_image_path))
