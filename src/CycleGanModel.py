# !/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "[Simon Audrix, Tarek Lammouchi, Quentin Lanneau, Gabriel Nativel-Fontaine]"
__date__ = "21-01-07"
__usage__ = "CycleGan model built following https://machinelearningmastery.com/cyclegan-tutorial-with-keras/"
__version__ = "1.0"

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, ReLU, Concatenate, Conv2DTranspose, Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from random import random
from numpy import asarray
from numpy.random import randint
import matplotlib.pyplot as plt


class CycleGan(Model):
    def __init__(self, shape):
        super(CycleGan, self).__init__(name='CycleGan')
        self._generator_AtoB = self.generator(shape)
        self._generator_BtoA = self.generator(shape)
        self._discriminatorA = self.discriminator(shape)
        self._discriminatorB = self.discriminator(shape)

        self.composite_AtoB = self.composite(self._generator_AtoB, self._discriminatorB, self._generator_BtoA, shape)
        self.composite_BtoA = self.composite(self._generator_BtoA, self._discriminatorA, self._generator_AtoB, shape)

    def discriminator(self, shape):
        init = RandomNormal(stddev=0.02)

        input_img = Input(shape=shape)
        x = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(input_img)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)

        patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(x)

        model = Model(input_img, patch_out)
        model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        return model

    def _resnet_block(self, n_filters, input_layer):
        init = RandomNormal(stddev=0.02)
        x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
        x = InstanceNormalization(axis=-1)(x)
        x = ReLU()(x)

        x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)

        x = Concatenate()([x, input_layer])
        return x

    def generator(self, image_shape, n_resnet=9):
        init = RandomNormal(stddev=0.02)

        in_image = Input(shape=image_shape)
        x = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
        x = InstanceNormalization(axis=-1)(x)
        x = ReLU()(x)

        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = ReLU()(x)

        x = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = ReLU()(x)

        for _ in range(n_resnet):
            x = self._resnet_block(256, x)

        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = ReLU()(x)

        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = ReLU()(x)

        x = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(x)
        x = InstanceNormalization(axis=-1)(x)
        out_image = Activation('tanh')(x)

        model = Model(in_image, out_image)
        return model

    def composite(self, g_model_1, d_model, g_model_2, image_shape):
        g_model_1.trainable = True
        d_model.trainable = False
        g_model_2.trainable = False

        input_gen = Input(shape=image_shape)
        gen1_out = g_model_1(input_gen)
        output_d = d_model(gen1_out)

        input_id = Input(shape=image_shape)
        output_id = g_model_1(input_id)

        output_f = g_model_2(gen1_out)

        gen2_out = g_model_2(input_id)
        output_b = g_model_1(gen2_out)

        model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
        return model

    def update_image_pool(self, pool, images, max_size=50):
        selected = list()
        for image in images:
            if len(pool) < max_size:
                # stock the pool
                pool.append(image)
                selected.append(image)
            elif random() < 0.5:
                # use image, but don't add it to the pool
                selected.append(image)
            else:
                # replace an existing image and use replaced image
                ix = randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return asarray(selected)

    def train(self, domainA, domainB, epochs=100, batch_size=1):
        train_summary_writer = tf.summary.create_file_writer('logs')

        poolA, poolB = list(), list()
        n_steps = len(domainA) // batch_size
        patch_shape = self._discriminatorA.output_shape[1]

        progress_bar_train = tf.keras.utils.Progbar(n_steps)

        # metrics to store
        dA_loss_1 = tf.keras.metrics.Mean('loss_discr_A_1')
        dA_loss_2 = tf.keras.metrics.Mean('loss_discr_A_2')
        dB_loss_1 = tf.keras.metrics.Mean('loss_discr_B_1')
        dB_loss_2 = tf.keras.metrics.Mean('loss_discr_B_2')
        cycle_loss_1 = tf.keras.metrics.Mean('cycle_loss_1')
        cycle_loss_2 = tf.keras.metrics.Mean('cycle_loss_2')

        for e in range(epochs):
            for step in range(n_steps):
                progress_bar_train.update(step)
                x_realA = domainA[step * batch_size:step * batch_size + batch_size]
                y_realA = tf.ones((batch_size, patch_shape, patch_shape, 1))

                x_realB = domainB[step * batch_size:step * batch_size + batch_size]
                y_realB = tf.ones((batch_size, patch_shape, patch_shape, 1))

                x_fakeA = self._generator_BtoA.predict(x_realB)
                y_fakeA = tf.zeros((len(x_fakeA), patch_shape, patch_shape, 1))

                x_fakeB = self._generator_AtoB.predict(x_realA)
                y_fakeB = tf.zeros((len(x_fakeB), patch_shape, patch_shape, 1))

                x_fakeA = self.update_image_pool(poolA, x_fakeA)
                x_fakeB = self.update_image_pool(poolB, x_fakeB)

                # update generator B->A via adversarial and cycle loss
                g_loss2, _, _, _, _ = self.composite_BtoA.train_on_batch([x_realB, x_realA],
                                                                         [y_realA, x_realA, x_realB, x_realA])
                # update discriminator for A -> [real/fake]
                dA_loss1 = self._discriminatorA.train_on_batch(x_realA, y_realA)
                dA_loss2 = self._discriminatorA.train_on_batch(x_fakeA, y_fakeA)

                # update generator A->B via adversarial and cycle loss
                g_loss1, _, _, _, _ = self.composite_AtoB.train_on_batch([x_realA, x_realB],
                                                                         [y_realB, x_realB, x_realA, x_realB])
                # update discriminator for B -> [real/fake]
                dB_loss1 = self._discriminatorB.train_on_batch(x_realB, y_realB)
                dB_loss2 = self._discriminatorB.train_on_batch(x_fakeB, y_fakeB)

                dA_loss_1.update_state(dA_loss1)
                dA_loss_2.update_state(dA_loss2)
                dB_loss_1.update_state(dB_loss1)
                dB_loss_2.update_state(dB_loss2)
                cycle_loss_1.update_state(g_loss1)
                cycle_loss_2.update_state(g_loss2)

            with train_summary_writer.as_default():
                tf.summary.scalar('dA_loss_1', dA_loss_1.result(), step=e)
                tf.summary.scalar('dA_loss_2', dA_loss_2.result(), step=e)
                tf.summary.scalar('dB_loss_1', dB_loss_1.result(), step=e)
                tf.summary.scalar('dB_loss_2', dB_loss_2.result(), step=e)
                tf.summary.scalar('cycle_loss_1', cycle_loss_1.result(), step=e)
                tf.summary.scalar('cycle_loss_2', cycle_loss_2.result(), step=e)

            s = f"Epoch {e}, losses:" \
                f"dA: [{dA_loss_1.result():.3f}, {dA_loss_2.result():.3f}]\n" \
                f"dB: [{dB_loss_1.result():.3f}, {dB_loss_2.result():.3f}]\n" \
                f"gA: [{cycle_loss_1.result():.3f}, {cycle_loss_2.result():.3f}]\n"

            print(s)
            self._generator_AtoB.save('models/generatorAtoB.h5')
            self._generator_BtoA.save('models/generatorBtoA.h5')

    def plot_performance(self, generator, images, name):
        fakes = generator.predict(images)

        # scale all pixels from [-1,1] to [0,1]
        X_in = (images + 1) / 2.0
        X_out = (fakes + 1) / 2.0

        # plot 5 real images
        for i in range(5):
            plt.subplot(2, 5, 1 + i)
            plt.axis('off')
            plt.imshow(X_in[i])

        # plot 5 translated image
        for i in range(5):
            plt.subplot(2, 5, 1 + 5 + i)
            plt.axis('off')
            plt.imshow(X_out[i])

        # save plot to file
        plt.savefig(f'{name}.png')
        plt.close()

