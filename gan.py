"""
wGAN implemented on top of keras/tensorflow as described in: [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
with improvements as described in: [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf).

"""

import os
import sys

from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

from dlutils import plot_image_batch_w_labels


# the names 'critic' and 'discriminator' are used interchangeably
# since the disc is no longer explicitly classifying input as real/generated (not required to output a probability)
# we refer to it as the 'critic'

#
# directory paths
#

path = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(path, 'cache')

#
# generator input params
#

rand_dim = 64  # dimension of the generator's input tensor (gaussian noise)

#
# image dimensions
#

img_height = 28
img_width = 28
img_channels = 3

#
# training params
#

nb_steps = 10000
batch_size = 64
k_d = 5  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
critic_pre_train_steps = 100  # number of steps to pre-train the critic before starting adversarial training

#
# logging params
#

log_interval = 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
fixed_noise = np.random.normal(size=(batch_size, rand_dim))  # fixed noise to generate batches of generated images

#
# shared network params
#

kernel_size = 3
conv_layer_keyword_args = {
    'strides': 2,
    'padding': 'same',
}


def generator_network(x):
    def add_common_layers(y):
        y = layers.Activation('relu')(y)
        return y

    x = layers.Dense(1024)(x)
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

    x = layers.Conv2DTranspose(64, kernel_size, **conv_layer_keyword_args)(x)
    x = add_common_layers(x)

    # number of feature maps => number of image channels
    return layers.Conv2DTranspose(img_channels, 1, strides=2, padding='same', activation='tanh')(x)


def discriminator_network(x):
    def add_common_layers(y):
        y = layers.advanced_activations.LeakyReLU()(y)
        return y

    x = layers.Conv2D(64, kernel_size, **conv_layer_keyword_args)(x)
    x = add_common_layers(x)

    x = layers.Conv2D(128, kernel_size, **conv_layer_keyword_args)(x)
    x = add_common_layers(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1024)(x)
    x = add_common_layers(x)

    return layers.Dense(1)(x)


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

    generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')

    # we need a second loss term for the gradient penalty
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output, discriminator_output],
                                       name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))[0]
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')

    #
    # define earth mover distance (wasserstein loss) and gradient penalty (improved wGAN)
    #

    # keras custom loss/objective functions are always minimized (=> small positive number or large negative number)
    def em_loss(y_coefficients, y_pred):
        return tf.reduce_mean(tf.multiply(y_coefficients, y_pred), axis=0)

    # kind of a hack to calculate the gradient penalty outside of the loss function
    # but this is the simplest way I could figure out to do this in keras
    def gradient_penalty(gradient_penalty_val, _):
        return gradient_penalty_val[0]

    #
    # compile models
    #

    adam = optimizers.Adam(lr=.0001, beta_1=0.5, beta_2=0.9)

    generator_model.compile(optimizer=adam, loss=[em_loss])
    discriminator_model.compile(optimizer=adam, loss=[em_loss, gradient_penalty])
    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss=[em_loss])

    print(generator_model.summary())
    print(discriminator_model.summary())
    print(combined_model.summary())

    #
    # data generators
    #

    data_generator = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        data_format='channels_last'
    )

    flow_from_directory_params = {
        'target_size': (img_height, img_width),
        'color_mode': 'grayscale' if img_channels == 1 else 'rgb',
        'class_mode': None,
        'batch_size': batch_size
    }

    real_image_generator = data_generator.flow_from_directory(
        directory=data_dir,
        **flow_from_directory_params,
    )

    def get_image_batch():
        img_batch = real_image_generator.next()

        # keras generators may generate an incomplete batch for the last batch in an epoch of data
        if len(img_batch) != batch_size:
            img_batch = real_image_generator.next()

        assert img_batch.shape == (batch_size, img_height, img_width, img_channels), img_batch.shape
        return img_batch

    disc_loss = []
    combined_loss = []

    #
    # build the computation graph for calculating the gradient penalty
    #

    # sample a batch of noise (generator input)
    _z = tf.random_normal(shape=(batch_size, rand_dim), mean=0.0, stddev=1.0, dtype=tf.float32)

    # sample a batch of real images
    _x = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, img_channels))

    # generate a batch of images with the current generator
    _g_z = generator_model(_z)

    # calculate `x_hat`
    epsilon = tf.random_uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0, dtype=tf.float32)
    x_hat = epsilon * _x + (1.0 - epsilon) * _g_z

    gradients = tf.gradients(discriminator_model(x_hat)[0], [x_hat], colocate_gradients_with_ops=True)
    _gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

    sess = K.get_session()
    sess.run(tf.global_variables_initializer())

    def train_discriminator_step():
        real_image_batch = get_image_batch()
        generated_image_batch, gp_loss = sess.run([_g_z, _gradient_penalty], feed_dict={_x: real_image_batch})

        dummy = np.zeros(batch_size * 2)
        dummy[0] = gp_loss

        # update φ by taking an SGD step on mini-batch loss LD(φ) TODO: no need to run through disc again
        return discriminator_model.train_on_batch(
            [np.concatenate((real_image_batch, generated_image_batch), axis=0)],
            [np.concatenate((-np.ones(batch_size), np.ones(batch_size)), axis=0), dummy]
        )

    if generator_model_path:
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        discriminator_model.load_weights(discriminator_model_path, by_name=True)
    else:
        print('pre-training the critic...')

        for i in range(critic_pre_train_steps):
            print('Step: {} of {} critic pre-training.'.format(i, critic_pre_train_steps))
            loss = train_discriminator_step()

        print('Last batch of critic pre-training disc_loss: {}.'.format(loss))
        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))

    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the discriminator
        for _ in range(k_d):
            # when plotting loss we will have to take `k_d` and `k_g` into account so the two plots align
            loss = train_discriminator_step()
            disc_loss.append(loss)

        # train the generator
        for _ in range(k_g):
            z = np.random.normal(size=(batch_size, rand_dim), loc=0.0, scale=1.0)

            # update θ by taking an SGD step on mini-batch loss LG(θ)
            loss = combined_model.train_on_batch(z, [-np.ones(batch_size)])
            combined_loss.append(loss)

        if not i % log_interval and i != 0:
            # plot batch of generated images w/ current generator
            figure_name = 'generated_image_batch_step_{}.png'.format(i)
            print('Saving batch of generated images at adversarial step: {}.'.format(i))

            g_z = generator_model.predict(fixed_noise)
            x = get_image_batch()

            # save one generated image to disc
            Image.fromarray(g_z[0], mode='RGB').save(os.path.join(cache_dir, 'generated_image_step_{}.png').format(i))
            # save a batch of generated and real images to disc
            plot_image_batch_w_labels.plot_batch(np.concatenate((g_z, x)), os.path.join(cache_dir, figure_name),
                                                 label_batch=['generated'] * batch_size + ['real'] * batch_size)

            # log loss summary
            print('Generator model loss: {}.'.format(np.mean(np.asarray(combined_loss[-log_interval:]), axis=0)))
            print('Discriminator model loss: {}.'.format(np.mean(np.asarray(disc_loss[-log_interval:]), axis=0)))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_weights_step_{}.h5')
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))


def main(data_dir, generator_model_path, discriminator_model_path):
    adversarial_training(data_dir, generator_model_path, discriminator_model_path)


if __name__ == '__main__':
    gen_model_path = sys.argv[2] if len(sys.argv) >= 3 else None
    disc_model_path = sys.argv[3] if len(sys.argv) >= 4 else None

    main(sys.argv[1], gen_model_path, disc_model_path)
