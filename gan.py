"""
Standard GAN implemented on top of keras/tensorflow.

Note: Currently only supports Python 3.
"""

import os
import sys

from keras import applications
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import numpy as np

from utils import plot_images


#
# directory paths
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')

# dimension of generator's input tensor
rand_dim = 112

#
# image dimensions
#

img_height = 112
img_width = 112
img_channels = 3

#
# training params
#

nb_steps = 10000
batch_size = 128
k_d = 1  # number of discriminator network updates per step
k_g = 2  # number of generative network updates per step
log_interval = 100  # interval (in steps) at which to log loss summaries & save plots of image samples to disc


#
# shared network params
#

kernel_size = (3, 3)
conv_layer_keyword_args = {'border_mode': 'same', 'subsample': (2, 2)}


def generator_network(input_tensor):
    def add_common_layers(y):
        y = layers.Activation('relu')(y)
        return y

    x = layers.Dense(1024)(input_tensor)
    x = add_common_layers(x)

    #
    # input dimensions to the first conv layer in the generator
    #

    height_dim = 7
    width_dim = 7
    nb_feature_maps = 128

    assert img_height % height_dim == 0 and img_width % width_dim == 0, \
        'Generator network must be able to transform `x` into a tensor of shape (img_height, img_width, img_channels).'

    x = layers.Dense(height_dim * width_dim * nb_feature_maps)(x)
    x = add_common_layers(x)

    x = layers.Reshape((height_dim, width_dim, nb_feature_maps))(x)

    nb_feature_maps = 512 * 2

    # generator will transform `x` into a tensor w/ the desired shape by up-sampling the spatial dimension of `x`
    # through a series of strided de-convolutions (each de-conv layer up-samples spatial dim of `x` by a factor of 2).
    while height_dim != img_height:
        height_dim *= 2  # (14 => 28 => 56 => 112 == img_height)
        width_dim *= 2
        nb_feature_maps //= 2  # (512 => 256 => 128 => 64)

        x = layers.convolutional.Deconvolution2D(nb_feature_maps, *kernel_size,
                                                 output_shape=(None, height_dim, width_dim, nb_feature_maps),
                                                 **conv_layer_keyword_args)(x)
        x = add_common_layers(x)

    # number of feature maps => number of image channels
    return layers.convolutional.Deconvolution2D(img_channels, *kernel_size, activation='tanh',
                                                border_mode='same',
                                                output_shape=(None, img_height, img_width, img_channels))(x)


def discriminator_network(input_image_tensor):
    def add_common_layers(y):
        y = layers.advanced_activations.LeakyReLU()(y)
        return y

    height_dim = 7

    x = input_image_tensor
    nb_feature_maps = 64 // 2

    # down sample with strided convolutions until we reach the desired spatial dimension
    while x.get_shape()[1] != height_dim:
        nb_feature_maps *= 2  # (64 => 128 => 256 => 512)

        x = layers.convolutional.Convolution2D(nb_feature_maps, *kernel_size, **conv_layer_keyword_args)(x)
        x = add_common_layers(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1024)(x)
    x = add_common_layers(x)

    return layers.Dense(1, activation='sigmoid')(x)


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

    combined_output = discriminator_network(generator_network(generator_input_tensor))

    #
    # define models
    #

    generator_model = models.Model(input=generator_input_tensor, output=generated_image_tensor, name='generator')
    discriminator_model = models.Model(input=generated_or_real_image_tensor, output=discriminator_output,
                                       name='discriminator')
    combined_model = models.Model(input=generator_input_tensor, output=combined_output, name='combined')

    #
    # compile models
    #

    sgd = optimizers.SGD(lr=0.001)

    generator_model.compile(optimizer=sgd, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    discriminator_model.trainable = False
    combined_model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    #
    # data generators
    #

    data_generator = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        dim_ordering='tf')

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

        assert len(img_batch) == batch_size
        return img_batch

    # the target labels for the binary cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (generated)
    y_real = np.array([0] * batch_size)
    y_generated = np.array([1] * batch_size)

    combined_loss = np.zeros(shape=len(combined_model.metrics_names))
    disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
    disc_loss_generated = np.zeros(shape=len(discriminator_model.metrics_names))

    if generator_model_path:
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        discriminator_model.load_weights(discriminator_model_path, by_name=True)

    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the discriminator
        for _ in range(k_d):
            generator_input = np.random.normal(size=(batch_size, rand_dim))
            # sample a mini-batch of real images
            real_image_batch = get_image_batch()

            # generate a batch of images with the current generator
            generated_image_batch = generator_model.predict(generator_input)

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss_real = np.add(discriminator_model.train_on_batch(real_image_batch, y_real), disc_loss_real)
            disc_loss_generated = np.add(discriminator_model.train_on_batch(generated_image_batch, y_generated),
                                         disc_loss_generated)

        # train the generator
        for _ in range(k_g * 2):
            generator_input = np.random.normal(size=(batch_size, rand_dim))

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            combined_loss = np.add(combined_model.train_on_batch(generator_input, y_real), combined_loss)

        if not i % log_interval and i != 0:
            # plot batch of generated images w/ current generator
            figure_name = 'generated_image_batch_step_{}.png'.format(i)
            print('Saving batch of generated images at adversarial step: {}.'.format(i))

            plot_images.plot_batch(generator_model.predict(np.random.normal(size=(batch_size, rand_dim))),
                                   get_image_batch(),
                                   os.path.join(cache_dir, figure_name))

            # log loss summary
            print('Generator model loss: {}.'.format(combined_loss / (log_interval * k_g * 2)))
            print('Discriminator model loss real: {}.'.format(disc_loss_real / (log_interval * k_d * 2)))
            print('Discriminator model loss generated: {}.'.format(disc_loss_generated / (log_interval * k_d * 2)))

            combined_loss = np.zeros(shape=len(combined_model.metrics_names))
            disc_loss_real = np.zeros(shape=len(discriminator_model.metrics_names))
            disc_loss_generated = np.zeros(shape=len(discriminator_model.metrics_names))

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
