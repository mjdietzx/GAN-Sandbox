"""
wGAN implemented on top of keras/tensorflow as described in: [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf).

Note: Requires Keras 2.0, `keras-2` branch, and Python 3.
"""

import os
import sys

import keras
assert int(keras.__version__[0]) == 2, 'Requires Keras v2.x.x. This corresponds to branch `keras-2`. ' \
                                       'Run `$ sudo pip3 install -U git+https://github.com/fchollet/keras.git@keras-2`.'

from keras import applications
from keras import constraints
from keras import layers
from keras import models
from keras import initializers
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

from dlutils import plot_image_batch_w_labels


# NOTE: the names `critic` and `discriminator` are used interchangeably
# since the disc is no longer explicitly classifying input as real/generated (not required to output a probability)
# we refer to it as the `critic`

#
# directory paths
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')

#
# generator input params
#

rand_dim = 112  # dimension of generator's input tensor (gaussian noise)

#
# image dimensions
#

img_height = 112
img_width = 112
img_channels = 3

#
# training params (as defined in Algorithm 1 of the paper)
#

nb_steps = 10000
batch_size = 64
k_d = 5  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
log_interval = 5  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
learning_rate = 0.00005
clipping_parameter = 0.01
critic_pre_train_steps = 100  # number of steps to pre-train the critic before starting adversarial training


#
# shared network params
#

def get_weight_initializer():
    return initializers.TruncatedNormal(mean=0.0, stddev=clipping_parameter / 2.0)


def get_weight_constraints(axis):
    return constraints.MinMaxNorm(min_value=-clipping_parameter, max_value=clipping_parameter, rate=1.0, axis=axis)

kernel_size = (3, 3)
conv_layer_keyword_args = {
    'padding': 'same',
    'strides': 2,
    'kernel_initializer': get_weight_initializer(),
    'kernel_constraint': get_weight_constraints([0, 1, 2, 3])
}
conv_layer_keyword_args_gen = {'padding': 'same', 'strides': 2}
dense_layer_keyword_args = {
    'kernel_initializer': get_weight_initializer(),
    'kernel_constraint': get_weight_constraints([0, 1])
}


def generator_network(input_tensor):
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        return y

    x = layers.Dense(1024)(input_tensor)
    x = add_common_layers(x)

    #
    # input dimensions to the first de-conv layer in the generator
    #

    height_dim = 7
    width_dim = 7
    assert img_height % height_dim == 0 and img_width % width_dim == 0, \
        'Generator network must be able to transform `x` into a tensor of shape (img_height, img_width, img_channels).'

    x = layers.Dense(height_dim * width_dim * 128)(x)
    x = add_common_layers(x)

    x = layers.Reshape((height_dim, width_dim, -1))(x)

    # generator will transform `x` into a tensor w/ the desired shape by up-sampling the spatial dimension of `x`
    # through a series of strided de-convolutions (each de-conv layer up-samples spatial dim of `x` by a factor of 2)
    while height_dim != img_height:
        # spatial dim: (14 => 28 => 56 => 112 == img_height == img_width)
        height_dim *= 2
        width_dim *= 2

        # nb_feature_maps: (512 => 256 => 128 => 64)
        try:
            nb_feature_maps //= 2
        except NameError:
            nb_feature_maps = 512

        x = layers.Deconvolution2D(nb_feature_maps, kernel_size,
                                   **conv_layer_keyword_args_gen)(x)
        x = add_common_layers(x)

    # number of feature maps => number of image channels
    return layers.Deconvolution2D(img_channels, (1, 1),
                                  activation='tanh',
                                  padding='same')(x)


def discriminator_network(x):
    def add_common_layers(y):
        # y = layers.BatchNormalization()(y)  # TODO: bug in `keras` causes exception when using bn in discriminator
        y = layers.advanced_activations.LeakyReLU()(y)
        return y

    height_dim = 7

    # down sample with strided convolutions until we reach the desired spatial dimension (7 * 7 * nb_feature_maps)
    while x.get_shape()[1] != height_dim:
        # nb_feature_maps: (64 => 128 => 256 => 512)
        try:
            nb_feature_maps *= 2
        except NameError:
            nb_feature_maps = 64

        x = layers.Convolution2D(nb_feature_maps, kernel_size, **conv_layer_keyword_args)(x)
        x = add_common_layers(x)

    x = layers.Flatten()(x)
    # x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(1024, **dense_layer_keyword_args)(x)
    x = add_common_layers(x)

    return layers.Dense(1, **dense_layer_keyword_args)(x)


def adversarial_training(data_dir, generator_model_path, discriminator_model_path):
    """
    Adversarial training of the generator network Gθ and discriminator network Dφ.

    """
    #
    # define model input and output tensors
    #

    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    generated_image_tensor = generator_network(generator_input_tensor)

    generated_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    discriminator_output = discriminator_network(generated_or_real_image_tensor)

    #
    # define models
    #

    generator_model = models.Model(inputs=generator_input_tensor, outputs=generated_image_tensor, name='generator')
    discriminator_model = models.Model(inputs=generated_or_real_image_tensor, outputs=discriminator_output,
                                       name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(inputs=generator_input_tensor, outputs=combined_output, name='combined')

    #
    # define earth mover distance (Wasserstein loss)
    #

    # NOTE: keras custom loss/objective functions are minimized
    def em_loss(y_coefficients, y_pred):
        # critic/discriminator: minimize E(f(g(z))) - E(f(x)) === E(f(g(z) - f(x)) === E(1 * f(g(z)) + -1 * f(x))
        # => minimize f(g(z) and maximize f(x)
        # generator: minimize -E(f(g(z))) === E(-1 * f(g(z)))
        # => maximize f(g(z)) (i.e. generate data the critic thinks is real)
        return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

    #
    # compile models
    #

    rms_prop = optimizers.RMSprop(lr=learning_rate)

    generator_model.compile(optimizer=rms_prop, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=rms_prop, loss=em_loss)
    discriminator_model.trainable = False
    combined_model.compile(optimizer=rms_prop, loss=em_loss)

    print(generator_model.summary())
    print(discriminator_model.summary())
    print(combined_model.summary())

    #
    # data generators
    #

    data_generator = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        data_format='channels_last')

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size}

    real_image_generator = data_generator.flow_from_directory(
        directory=data_dir,
        **flow_from_directory_params
    )

    def get_image_batch():
        img_batch = real_image_generator.next()

        # keras generators may generate an incomplete batch for the last batch
        if len(img_batch) != batch_size:
            img_batch = real_image_generator.next()

        assert img_batch.shape == (batch_size, img_height, img_width, img_channels), img_batch.shape
        return img_batch

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

    def train_discriminator_step():
        # sample a mini-batch of noise (generator input)
        z = np.random.normal(size=(batch_size, rand_dim))

        # sample a mini-batch of real images
        x = get_image_batch()

        # generate a batch of images with the current generator
        g_z = generator_model.predict(z)

        x = np.concatenate((g_z, x))
        assert x.shape == (batch_size * 2, img_height, img_width, img_channels)

        # coefficients used to compute earth mover loss (not ground-truth labels)
        y = np.concatenate((np.ones(batch_size), -np.ones(batch_size)))

        # update φ by taking an SGD step on mini-batch loss LD(φ)
        return discriminator_model.train_on_batch(x, y)

    if generator_model_path:
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        discriminator_model.load_weights(discriminator_model_path, by_name=True)
    else:
        # pre-train the critic as described in: https://github.com/martinarjovsky/WassersteinGAN so adversarial training
        # starts with the critic near optimum (otherwise loss would go up until critic is properly trained)
        print('pre-training the discriminator network...')

        for i in range(critic_pre_train_steps):
            print('Step: {} of {} critic pre-train.'.format(i, nb_steps))
            disc_loss = np.add(train_discriminator_step(), disc_loss)

        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))
        print('Discriminator model loss: {}.'.format(disc_loss / critic_pre_train_steps))

    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the discriminator
        for _ in range(k_d):
            disc_loss = np.add(train_discriminator_step(), disc_loss)

        # train the generator
        for _ in range(k_g):
            generator_input = np.random.normal(size=(batch_size, rand_dim))

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(combined_model.train_on_batch(generator_input, -np.ones(batch_size)),
                                   combined_loss)

        if not i % log_interval and i != 0:
            # plot batch of generated images w/ current generator
            figure_name = 'generated_image_batch_step_{}.png'.format(i)
            print('Saving batch of generated images at adversarial step: {}.'.format(i))

            generated_image_batch = generator_model.predict(np.random.normal(size=(batch_size, rand_dim)))
            real_image_batch = get_image_batch()

            plot_image_batch_w_labels.plot_batch(np.concatenate((generated_image_batch, real_image_batch)),
                                                 os.path.join(cache_dir, figure_name),
                                                 label_batch=['generated'] * batch_size + ['real'] * batch_size)

            # log loss summary
            print('Generator model loss: {}.'.format(combined_loss / (log_interval * k_g)))
            print('Discriminator model loss: {}.'.format(disc_loss / (log_interval * k_d)))

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss = np.zeros(shape=len(discriminator_model.metrics_names))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_weights_step_{}.h5')
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))


def main(data_dir, generator_model_path, discriminator_model_path):
    adversarial_training(data_dir, generator_model_path, discriminator_model_path)


if __name__ == '__main__':
    # Note: if pre-trained models are passed in we don't take the steps they've been trained for into account
    gen_model_path = sys.argv[2] if len(sys.argv) >= 3 else None
    disc_model_path = sys.argv[3] if len(sys.argv) >= 4 else None

    main(sys.argv[1], gen_model_path, disc_model_path)
