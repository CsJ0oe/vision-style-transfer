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


def transfer_style(nb_epochs, content, style, style_layer_names, content_layer_name, name):
    """
    :param nb_epochs: How many epochs
    :param content: content image
    :param style: style image
    :param style_layer_names: layers to use for style loss
    :param content_layer_name: layers to use for content loss
    :param name:
    :return:
    """

    # Download content and style images
    if not isinstance(content, np.ndarray):
        base_image_path = get_file(f"{int(time.time())}", content)
        width, height = load_img(base_image_path).size
    else:
        base_image_path = content
        width, height, _ = content.shape

    if not isinstance(style, np.ndarray):
        style_reference_image_path = get_file(f"{int(time.time())}", style)
    else:
        style_reference_image_path = style

    """### Utility functions"""

    def preprocess_image(image_path, height, width):
        """Open, resize and format a picture into appropriate tensors"""
        if not isinstance(image_path, np.ndarray):
            img = load_img(
                image_path, target_size=(height, width)
            )
            img = img_to_array(img)
        else:
            img = image_path
            img = resize(img, (height, width))

        img = np.expand_dims(img, axis=0)
        # Convert image from RGB to BGR and zero-center each color channel w.r.t. the ImageNet dataset
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)

    def deprocess_image(x, height, width):
        """Convert a tensor into a valid image"""
        x = x.reshape((height, width, 3))
        # Remove zero-center by mean pixel applied by vgg19.preprocess_input()
        # The following values are the mean pixel values of each color channel for the ImageNet dataset
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype("uint8")
        return x

    """### Loss functions"""

    # The gram matrix of a 3D tensor (correlations between the feature maps of a convolutional layer)
    def gram_matrix(x):
        # Transpose feature maps tensor tensor so that 3rd dimension becomes 1st
        x = tf.transpose(x, (2, 0, 1))
        # Reshape feature maps tensor into a matrix. First dimension is the number of filters/maps
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        # Compute the outer product of feature vectors with themselves
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    # The style loss is designed to maintain the style of the reference image in the generated image
    # It is based on the gram matrices (which capture style) of feature maps from the style reference image
    # and from the generated image
    def style_loss(style, combination, height, width):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = height * width
        # Compute distance between Gram matrices of style and generated images
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    # The content loss is designed to maintain the "content" of the base image in the generated image
    def content_loss(base, combination):
        return tf.reduce_sum(tf.square(combination - base))

    # The total variation loss is designed to keep the generated image locally coherent
    def total_variation_loss(x, height, width):
        a = tf.square(
            x[:, : height - 1, : width - 1, :] - x[:, 1:, : width - 1, :]
        )
        b = tf.square(
            x[:, : height - 1, : width - 1, :] - x[:, : height - 1, 1:, :]
        )
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    """### Model definition

    We use a [VGG](https://arxiv.org/abs/1409.1556) model pretrained on the ImageNet dataset.
    """

    # Using the convolutional base of VGG19, loaded with pre-trained ImageNet weights
    vgg = vgg19.VGG19(weights="imagenet", include_top=False)

    # Get the symbolic outputs of each "key" layer (we gave them unique names)
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])

    # Set up a model that returns the activation values for every layer in VGG19 (as a dict)
    feature_extractor = Model(inputs=vgg.inputs, outputs=outputs_dict)

    # vgg.summary()

    """### Loss computation"""
    # Weights of the different loss components
    total_variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8

    def compute_loss(combination_image, base_image, style_reference_image, height, width):
        input_tensor = tf.concat(
            [base_image, style_reference_image, combination_image], axis=0
        )
        features = feature_extractor(input_tensor)

        # Initialize the loss
        loss = tf.zeros(shape=())

        # Add content loss
        layer_features = features[content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + content_weight * content_loss(
            base_image_features, combination_features
        )
        # Add style loss
        for layer_name in style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features, height, width)
            loss += (style_weight / len(style_layer_names)) * sl

        # Add total variation loss
        loss += total_variation_weight * total_variation_loss(combination_image, height, width)
        return loss

    @tf.function
    def compute_loss_and_grads(combination_image, base_image, style_reference_image, height, width):
        with tf.GradientTape() as tape:
            loss = compute_loss(combination_image, base_image, style_reference_image, height, width)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    """### Training loop"""
    # Generated image height
    gen_height = 400
    # Compute generated width so that input and generated images have same scale
    gen_width = int(width * gen_height / height)
    # print(f"Generated image dimensions: {gen_width, gen_height}")

    optimizer = SGD(
        ExponentialDecay(
            initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
        )
    )

    base_image = preprocess_image(base_image_path, gen_height, gen_width)
    style_reference_image = preprocess_image(style_reference_image_path, gen_height, gen_width)
    combination_image = tf.Variable(preprocess_image(base_image_path, gen_height, gen_width))

    # Training loop
    n_epochs = nb_epochs
    for epoch in range(1, n_epochs + 1):
        loss, grads = compute_loss_and_grads(
            combination_image, base_image, style_reference_image, gen_height, gen_width
        )
        optimizer.apply_gradients([(grads, combination_image)])
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{n_epochs}], loss: {loss:.2f}")

    # Save final image
    final_img = deprocess_image(combination_image.numpy(), gen_height, gen_width)
    result_image_path = f"img_{name}_{n_epochs}.png"
    save_img(result_image_path, final_img)

    """### Generated image display"""

    # Show final generated image
    display(Image(result_image_path))
