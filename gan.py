"""
Conditional GAN (cGAN) implemented on top of keras/tensorflow.
See: Image-to-Image Translation with Conditional Adversarial Networks: https://arxiv.org/pdf/1611.07004v1.pdf

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
import tensorflow as tf

from utils import plot_images


#
# directory paths
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')

#
# generator input params
#

rand_dim = 112  # dimension of generator's input tensor

#
# image dimensions
#

img_height = 256
img_width = 256
img_channels = 3

#
# training params
#

nb_steps = 10000
batch_size = 4
k_d = 1  # number of discriminator network updates per step
k_g = 1  # number of generative network updates per step
log_interval = 100  # interval (in steps) at which to log loss summaries & save plots of image samples to disc


#
# shared network params
#

kernel_size = (4, 4)  # all convolutions are 4 x 4 spatial features
conv_layer_keyword_args = {'border_mode': 'same', 'subsample': (2, 2)}  # applied w/ stride 2


def add_common_layers(y, batchnorm=True, dropout=True):
    if batchnorm:
        y = layers.BatchNormalization()(y)
    if dropout:
        y = layers.Dropout(0.5)(y)

    return layers.advanced_activations.LeakyReLU(alpha=0.2)(y)


def generator_network(input_tensor):
    # encoder: C64-C128-C256-C512-C512-C512-C512-C512
    n_0 = layers.Convolution2D(64, *kernel_size, **conv_layer_keyword_args)(input_tensor)
    n_0 = add_common_layers(n_0, batchnorm=False, dropout=False)

    n_1 = layers.Convolution2D(128, *kernel_size, **conv_layer_keyword_args)(n_0)
    n_1 = add_common_layers(n_1, dropout=False)

    n_2 = layers.Convolution2D(256, *kernel_size, **conv_layer_keyword_args)(n_1)
    n_2 = add_common_layers(n_2, dropout=False)

    n_3 = layers.Convolution2D(512, *kernel_size, **conv_layer_keyword_args)(n_2)
    n_3 = add_common_layers(n_3, dropout=False)

    n_4 = layers.Convolution2D(512, *kernel_size, **conv_layer_keyword_args)(n_3)
    n_4 = add_common_layers(n_4, dropout=False)

    n_5 = layers.Convolution2D(512, *kernel_size, **conv_layer_keyword_args)(n_4)
    n_5 = add_common_layers(n_5, dropout=False)

    n_6 = layers.Convolution2D(512, *kernel_size, **conv_layer_keyword_args)(n_5)
    n_6 = add_common_layers(n_6, dropout=False)

    n_7 = layers.Convolution2D(512, *kernel_size, **conv_layer_keyword_args)(n_6)
    n_7 = add_common_layers(n_7, dropout=False)

    # decoder: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    x = layers.Deconvolution2D(512, *kernel_size, **conv_layer_keyword_args, output_shape=(None, 2, 2, 512))(n_7)
    x = add_common_layers(x)
    x = layers.merge([n_6, x], mode='concat', concat_axis=3)

    x = layers.Deconvolution2D(512, *kernel_size, **conv_layer_keyword_args, output_shape=(None, 4, 4, 512))(x)
    x = add_common_layers(x)
    x = layers.merge([n_5, x], mode='concat', concat_axis=3)

    x = layers.Deconvolution2D(512, *kernel_size, **conv_layer_keyword_args, output_shape=(None, 8, 8, 512))(x)
    x = add_common_layers(x)
    x = layers.merge([n_4, x], mode='concat', concat_axis=3)

    x = layers.Deconvolution2D(512, *kernel_size, **conv_layer_keyword_args, output_shape=(None, 16, 16, 512))(x)
    x = add_common_layers(x, dropout=False)
    x = layers.merge([n_3, x], mode='concat', concat_axis=3)

    x = layers.Deconvolution2D(256, *kernel_size, **conv_layer_keyword_args, output_shape=(None, 32, 32, 256))(x)
    x = add_common_layers(x, dropout=False)
    x = layers.merge([n_2, x], mode='concat', concat_axis=3)

    x = layers.Deconvolution2D(128, *kernel_size, **conv_layer_keyword_args, output_shape=(None, 64, 64, 128))(x)
    x = add_common_layers(x, dropout=False)
    x = layers.merge([n_1, x], mode='concat', concat_axis=3)

    x = layers.Deconvolution2D(64, *kernel_size, **conv_layer_keyword_args, output_shape=(None, 128, 128, 64))(x)
    x = add_common_layers(x, dropout=False)
    x = layers.merge([n_0, x], mode='concat', concat_axis=3)

    # number of feature maps => number of image channels
    return layers.Deconvolution2D(img_channels, *kernel_size, **conv_layer_keyword_args, activation='tanh',
                                  output_shape=(None, img_height, img_width, img_channels))(x)


# NOTE: `x` is two input images concatenated: (edges, image) => (img_height, img_width, img_channels * 2)
def discriminator_network(x):
    # The 70 × 70 discriminator architecture is: C64-C128-C256-C512
    x = layers.Convolution2D(64, *kernel_size, **conv_layer_keyword_args)(x)
    x = add_common_layers(x, batchnorm=False, dropout=False)

    x = layers.Convolution2D(128, *kernel_size, **conv_layer_keyword_args)(x)
    x = add_common_layers(x, dropout=False)

    x = layers.Convolution2D(256, *kernel_size, **conv_layer_keyword_args)(x)
    x = add_common_layers(x, dropout=False)

    x = layers.Convolution2D(512, *kernel_size, **conv_layer_keyword_args)(x)
    x = add_common_layers(x, dropout=False)

    # NOTE: using softmax (applied in loss) instead of sigmoid as described in paper...
    return layers.Convolution2D(2, 1, 1, border_mode='same')(x)


def adversarial_training(data_dir, generator_model_path, discriminator_model_path):
    """
    Adversarial training of the generator network Gθ(x, z) => y and discriminator network Dφ(x, y) => real/fake.

    """
    #
    # define model input and output tensors
    #

    # G takes observed image x as input (noise is provided only in the form of dropout - applied at both train/test)
    generator_input_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    generated_image_tensor = generator_network(generator_input_tensor)

    # unlike an unconditional GAN, both the generator and discriminator observe an input image
    generated_or_real_image_pair_tensor = layers.Input(shape=(img_height, img_width, img_channels * 2))
    discriminator_output = discriminator_network(generated_or_real_image_pair_tensor)

    # combined must output the generated image along w/ the disc's classification for the generator's self-reg loss
    combined_output = discriminator_network(layers.merge(
        (generator_input_tensor, generator_network(generator_input_tensor)), mode='concat', concat_axis=3))

    #
    # define models
    #

    generator_model = models.Model(input=generator_input_tensor, output=generated_image_tensor, name='generator')
    discriminator_model = models.Model(input=generated_or_real_image_pair_tensor, output=discriminator_output,
                                       name='discriminator')
    combined_model = models.Model(input=generator_input_tensor, output=[generated_image_tensor, combined_output],
                                  name='combined')

    discriminator_model_output_shape = discriminator_model.output_shape

    # define custom l1 loss function for the generator
    def self_regularization_loss(y_true, y_pred):
        delta = 100.0
        return tf.multiply(delta, tf.reduce_sum(tf.abs(y_pred - y_true)))

    # define custom local adversarial loss (softmax for each image section) for the discriminator
    # the adversarial loss function is the sum of the cross-entropy losses over the local patches
    def local_adversarial_loss(y_true, y_pred):
        # y_true and y_pred have shape (batch_size, # of local patches, 2), but really we just want to average over
        # the local patches and batch size so we can reshape to (batch_size * # of local patches, 2)
        y_true = tf.reshape(y_true, (-1, 2))
        y_pred = tf.reshape(y_pred, (-1, 2))
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        return tf.reduce_mean(loss)

    #
    # compile models
    #

    adam = optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)  # as described in appendix A of DeepMind's AC-GAN paper

    generator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=adam, loss=local_adversarial_loss)
    discriminator_model.trainable = False
    # the generator is tasked to not only fool the discriminator but also to be near the ground truth output (L1)
    combined_model.compile(optimizer=adam, loss=[self_regularization_loss, local_adversarial_loss])

    print(generator_model.summary())
    print(discriminator_model.summary())

    #
    # data generators
    #

    data_generator = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        dim_ordering='tf')

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size,
                                  # real and synthesized pairs must be coordinated
                                  'shuffle': False}

    train_dirs = []
    for directory in os.listdir(data_dir):
        directory_path = os.path.join(data_dir, directory)
        if directory.startswith('.') or not os.path.isdir(directory_path):
            continue
        train_dirs.append(directory_path)

    # TODO: should edges be rgb or grayscale?
    edge_image_generator = data_generator.flow_from_directory(
        directory=train_dirs,
        classes=['melanoma_edges', 'non_melanoma_edges'],
        **flow_from_directory_params
    )

    real_image_generator = data_generator.flow_from_directory(
        directory=train_dirs,
        classes=['melanoma', 'non_melanoma'],
        **flow_from_directory_params
    )

    def get_image_pair_batch():
        img_batch = real_image_generator.next()
        edge_img_batch = edge_image_generator.next()

        # keras generators may generate an incomplete batch for the last batch
        if len(img_batch) != batch_size:
            img_batch = real_image_generator.next()
            edge_img_batch = edge_image_generator.next()

        assert img_batch.shape == (batch_size, img_height, img_width, img_channels), img_batch.shape
        assert edge_img_batch.shape == (batch_size, img_height, img_width, img_channels), edge_img_batch.shape

        return img_batch, edge_img_batch

    # the target labels for the cross-entropy loss layer are 0 for every yj (real) and 1 for every xi (generated)
    y_real = np.array([[[[1.0, 0.0]] * discriminator_model_output_shape[1]] * discriminator_model_output_shape[2]]
                      * batch_size)
    y_generated = np.array([[[[0.0, 1.0]] * discriminator_model_output_shape[1]] * discriminator_model_output_shape[2]]
                           * batch_size)
    assert y_real.shape == (batch_size, ) + discriminator_model_output_shape[1:3] + (2, ), y_real.shape

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
            # sample a mini-batch of real image pairs
            image_batch, edge_image_batch = get_image_pair_batch()
            disc_input_real = np.concatenate((edge_image_batch, image_batch), axis=3)

            # update φ by taking an SGD step on mini-batch loss LD(φ)
            disc_loss_real = np.add(discriminator_model.train_on_batch(disc_input_real, y_real), disc_loss_real)

            # sample a mini-batch of edge images to be used to generate an image_batch
            _, edge_image_batch = get_image_pair_batch()

            # generate a batch of images with the current generator
            generated_image_batch = generator_model.predict(edge_image_batch)
            disc_input_generated = np.concatenate((edge_image_batch, generated_image_batch), axis=3)

            disc_loss_generated = np.add(discriminator_model.train_on_batch(disc_input_generated, y_generated),
                                         disc_loss_generated)

        # train the generator
        for _ in range(k_g * 2):
            _, edge_image_batch = get_image_pair_batch()

            combined_loss = np.add(combined_model.train_on_batch(edge_image_batch, [edge_image_batch, y_real]),
                                   combined_loss)

        if not i % log_interval:
            # plot batch of generated images w/ current generator
            figure_name = 'generated_image_batch_step_{}.png'.format(i)
            print('Saving batch of generated images at adversarial step: {}.'.format(i))

            image_batch, edge_image_batch = get_image_pair_batch()

            g = generator_model.predict(edge_image_batch)
            print(g.shape)
            plot_images.plot_batch(g,
                                   image_batch,
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
