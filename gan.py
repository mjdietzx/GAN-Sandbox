"""
wGAN implemented on top of keras/tensorflow as described in: [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf).

"""

import os
import pickle
import sys

from keras import applications
from keras import layers
from keras import models
from keras import initializers
from keras import optimizers
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

from dlutils import plot_image_batch_w_labels


# NOTE: the names 'critic' and 'discriminator' are used interchangeably
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
# training params (as defined in Algorithm 1 of the paper)
#

nb_steps = 10000
batch_size = 64
k_d = 5  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
learning_rate = 0.0005
clipping_parameter = 0.1
critic_pre_train_steps = 10  # number of steps to pre-train the critic before starting adversarial training

#
# logging params
#

log_interval = 10  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
fixed_noise = np.random.normal(size=(batch_size, rand_dim))  # fixed noise to generate batches of generated images


#
# shared network params
#

def get_weight_initializer():
    return initializers.TruncatedNormal(mean=0.0, stddev=clipping_parameter / 2.0)

kernel_size = 3
conv_layer_keyword_args = {
    'strides': 2,
    'padding': 'same',
    'kernel_initializer': get_weight_initializer(),
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

    #
    # define models
    #

    generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor], outputs=[discriminator_output],
                                       name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')

    #
    # define earth mover distance (Wasserstein loss)
    #

    # keras custom loss/objective functions are always minimized (=> small positive number or large negative number)
    def em_loss(y_coefficients, y_pred):
        # critic/discriminator: minimize f(x) and maximize f(g(z)) (minimize -f(g(z)))
        # generator: minimize f(g(z))
        return tf.reduce_mean(tf.multiply(y_coefficients, y_pred), axis=0)

    #
    # compile models
    #

    rms_prop = optimizers.RMSprop(lr=learning_rate)

    generator_model.compile(optimizer=rms_prop, loss=em_loss)
    discriminator_model.compile(optimizer=rms_prop, loss=[em_loss])
    discriminator_model.trainable = False
    combined_model.compile(optimizer=rms_prop, loss=[em_loss])

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

        # keras generators may generate an incomplete batch for the last batch in an epoch of data
        if len(img_batch) != batch_size:
            img_batch = real_image_generator.next()

        assert img_batch.shape == (batch_size, img_height, img_width, img_channels), img_batch.shape
        return img_batch

    def clamp_weights():
        for layer in discriminator_model.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -clipping_parameter, clipping_parameter) for w in weights]
            layer.set_weights(weights)

    disc_loss_real = np.empty(shape=(1, ))
    disc_loss_generated = np.empty(shape=(1, ))
    combined_loss = np.empty(shape=(1, ))

    def train_discriminator_step():
        # sample a batch of noise (generator input)
        z = np.random.normal(size=(batch_size, rand_dim))

        # sample a batch of real images
        x = get_image_batch()

        # generate a batch of images with the current generator
        g_z = generator_model.predict(z)

        # update φ by taking an SGD step on mini-batch loss LD(φ)
        l_r = discriminator_model.train_on_batch(x, np.ones(batch_size))
        l_g = discriminator_model.train_on_batch(g_z, -np.ones(batch_size))

        # clamp parameters to a cube
        clamp_weights()

        return l_r, l_g

    if generator_model_path:
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        discriminator_model.load_weights(discriminator_model_path, by_name=True)
    else:
        # pre-train the critic as described in: https://github.com/martinarjovsky/WassersteinGAN so adversarial training
        # starts with the critic near optimum (otherwise loss would go up until critic is properly trained)
        print('pre-training the critic (discriminator network)...')

        for i in range(critic_pre_train_steps):
            print('Step: {} of {} critic pre-train.'.format(i, critic_pre_train_steps))
            loss_real, loss_generated = train_discriminator_step()

        print('Last batch of critic pre-training disc_loss_real: {}, disc_loss_generated: {}.'.format(loss_real,
                                                                                                      loss_generated))
        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))

    for i in range(nb_steps):
        print('Step: {} of {}.'.format(i, nb_steps))

        # train the discriminator
        for _ in range(k_d):
            loss_real, loss_generated = train_discriminator_step()

        # only save loss for the last step in `k_d`
        disc_loss_real = np.append(disc_loss_real, loss_real)
        disc_loss_generated = np.append(disc_loss_generated, loss_generated)

        # train the generator
        for _ in range(k_g):
            z = np.random.normal(size=(batch_size, rand_dim))

            # update θ by taking an SGD step on mini-batch loss LR(θ)
            loss_combined = combined_model.train_on_batch(z, [np.ones(batch_size, dtype=np.float32)])

        # only save loss for the last step in `k_g`
        combined_loss = np.append(combined_loss, loss_combined)

        if not i % log_interval and i != 0:
            # plot batch of generated images w/ current generator
            figure_name = 'generated_image_batch_step_{}.png'.format(i)
            print('Saving batch of generated images at adversarial step: {}.'.format(i))

            g_z = generator_model.predict(fixed_noise)
            x = get_image_batch()

            plot_image_batch_w_labels.plot_batch(np.concatenate((g_z, x)), os.path.join(cache_dir, figure_name),
                                                 label_batch=['generated'] * batch_size + ['real'] * batch_size)

            # log loss summary
            print('Generator model loss: {}.'.format(np.mean(combined_loss[-log_interval:], axis=0)))
            print('Discriminator model loss real: {}.'.format(np.mean(disc_loss_real[-log_interval:], axis=0)))
            print('Discriminator model loss generated: {}.'.format(np.mean(disc_loss_generated[-log_interval:], axis=0)))

            # save model checkpoints
            model_checkpoint_base_name = os.path.join(cache_dir, '{}_model_weights_step_{}.h5')
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))

            # write the losses to disc as a serialized dict
            with open(os.path.join(cache_dir, 'losses.pickle'), 'wb') as handle:
                pickle.dump({'combined_loss': combined_loss,
                             'disc_loss_real': disc_loss_real,
                             'disc_loss_generated': disc_loss_generated},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(data_dir, generator_model_path, discriminator_model_path):
    adversarial_training(data_dir, generator_model_path, discriminator_model_path)


if __name__ == '__main__':
    # Note: if pre-trained models are passed in we don't take the steps they've been trained for into account
    gen_model_path = sys.argv[2] if len(sys.argv) >= 3 else None
    disc_model_path = sys.argv[3] if len(sys.argv) >= 4 else None

    main(sys.argv[1], gen_model_path, disc_model_path)
