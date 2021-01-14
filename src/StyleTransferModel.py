#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "[Simon Audrix, Tarek Lammouchi, Quentin Lanneau, Gabriel Nativel-Fontaine]"
__date__ = "20-12-17"
__usage__ = "Neural style transfer algorithm from Gaty redesigned from https://github.com/bpesquet/mlhandbook"
__version__ = "2.0"


import tensorflow as tf
from IPython.display import Image, display
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg19
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import save_img

from src.utils import deprocess_image


class TransferStyle(Model):
    def __init__(self, style_layers, content_layers):
        super(TransferStyle, self).__init__(name='TransferStyle')

        self._total_variation_weight = 1e-6
        self._style_weight = 1e-6
        self._content_weight = 2.5e-8

        vgg = vgg19.VGG19(weights="imagenet", include_top=False)
        outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])
        self._outputs = [layer.name for layer in vgg.layers]
        self._feature_extractor = Model(inputs=vgg.inputs, outputs=outputs_dict)

        self._style_layers = style_layers
        self._content_layers = content_layers

        self._optimizer = SGD(ExponentialDecay(initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))
        # self._optimizer = Adam(0.001)

    def gram_matrix(self, x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    def style_loss(self, style, combination, height, width):
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        size = height * width
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    def content_loss(self, base, combination):
        return tf.reduce_sum(tf.square(combination - base))

    def total_variation_loss(self, x, height, width):
        a = tf.square(x[:, : height - 1, : width - 1, :] - x[:, 1:, : width - 1, :])
        b = tf.square(x[:, : height - 1, : width - 1, :] - x[:, : height - 1, 1:, :])
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    def compute_loss(self, combination_image, base_image, style_reference_image, height, width):
        input_tensor = tf.concat([base_image, style_reference_image, combination_image], axis=0)
        features = self._feature_extractor(input_tensor)

        loss = tf.zeros(shape=())

        # Content loss
        for l in self._content_layers:
            layer_features = features[self._outputs[l]]
            base_image_features = layer_features[0, :, :, :]
            combination_features = layer_features[2, :, :, :]
            loss += (self._content_weight / len(self._content_layers)) * self.content_loss(base_image_features, combination_features)

        # Style loss
        for l in self._style_layers:
            layer_features = features[self._outputs[l]]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self.style_loss(style_reference_features, combination_features, height, width)
            loss += (self._style_weight / len(self._style_layers)) * sl

        loss += self._total_variation_weight * self.total_variation_loss(combination_image, height, width)
        return loss

    @tf.function
    def compute_loss_and_grads(self, combination_image, base_image, style_reference_image, height, width):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(combination_image, base_image, style_reference_image, height, width)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    def fit(self, epochs, base_image, style_reference_image, name):
        combination_image = tf.Variable(base_image)
        gen_height = 400
        _, height, width, _ = base_image.shape
        gen_width = int(width * gen_height / height)

        for epoch in range(1, epochs + 1):
            loss, grads = self.compute_loss_and_grads(combination_image, base_image, style_reference_image, gen_height, gen_width)
            self._optimizer.apply_gradients([(grads, combination_image)])
            if epoch % 1 == 0:
                print(f"Epoch [{epoch}/{epochs}], loss: {loss:.2f}")

        # Save final image
        final_img = deprocess_image(combination_image.numpy(), gen_height, gen_width)
        result_image_path = f"img/img_{name}_{epochs}.png"
        save_img(result_image_path, final_img)

        display(Image(result_image_path))
