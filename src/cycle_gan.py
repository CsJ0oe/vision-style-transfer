from random import random

import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, ReLU, Concatenate, Conv2DTranspose, Activation
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from numpy import load, asarray
from tensorflow_addons.layers import InstanceNormalization
from numpy.random import randint


def discriminator(shape):
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


def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(input_layer)
    x = InstanceNormalization(axis=-1)(x)
    x = ReLU()(x)
    
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)
    
    x = Concatenate()([x, input_layer])
    return x


def generator(image_shape, n_resnet=9):
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
        x = resnet_block(256, x)

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


def load_real_samples(filename):
    # load the dataset
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
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


def generate_real_samples(dataset, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = tf.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y


def generate_fake_samples(g_model, dataset, patch_shape):
    X = g_model.predict(dataset)
    y = tf.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def update_image_pool(pool, images, max_size=50):
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


def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    n_epochs, n_batch, = 100, 1
    n_patch = d_model_A.output_shape[1]
    trainA, trainB = dataset
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)


        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        # summarize performance
        print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
        # evaluate the model performance every so often
        if (i+1) % (bat_per_epo * 1) == 0:
            # plot A->B translation
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B->A translation
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
        if (i+1) % (bat_per_epo * 5) == 0:
            # save the models
            save_models(i, g_model_AtoB, g_model_BtoA)


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
    # select a sample of input images
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    # generate translated images
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    # plot real images
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_in[i])
    # plot translated image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_out[i])
    # save plot to file
    filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
    pyplot.savefig(filename1)
    pyplot.close()


def save_models(step, g_model_AtoB, g_model_BtoA):
    # save the first generator model
    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))