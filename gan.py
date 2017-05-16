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

import ResNeXt


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

img_height = 56
img_width = 56
img_channels = 3

#
# training params
#

nb_steps = 50000
batch_size = 64
k_d = 5  # number of critic network updates per adversarial training step
k_g = 1  # number of generator network updates per adversarial training step
critic_pre_train_steps = 1  # number of steps to pre-train the critic before starting adversarial training

#
# logging params
#

log_interval = 100  # interval (in steps) at which to log loss summaries and save plots of image samples to disc
fixed_noise = np.random.normal(size=(batch_size, rand_dim))  # fixed noise to generate batches of generated images


def adversarial_training(data_dir, generator_model_path, discriminator_model_path, encoder_model_path):
    """
    Adversarial training of the generator network Gθ and discriminator network Dφ.

    """
    #
    # define model input and output tensors
    #

    latent_noise_tensor = layers.Input(shape=(rand_dim, ))
    generated_image_tensor = ResNeXt.decoder_network(latent_noise_tensor)

    real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    encoded_latent_noise_tensor = ResNeXt.encoder_network(real_image_tensor)

    encoded_or_real_latent_noise_tensor = layers.Input(shape=(rand_dim, ))
    generated_or_real_image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    discriminator_output = ResNeXt.discriminator_network(generated_or_real_image_tensor,
                                                         encoded_or_real_latent_noise_tensor)

    #
    # define models
    #

    generator_model = models.Model(inputs=[latent_noise_tensor], outputs=[generated_image_tensor], name='generator')
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor, encoded_or_real_latent_noise_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    encoder_model = models.Model(inputs=[real_image_tensor], outputs=[encoded_latent_noise_tensor], name='encoder')

    combined_output_g = discriminator_model(inputs=[generator_model(latent_noise_tensor), latent_noise_tensor])
    combined_model_g = models.Model(inputs=[latent_noise_tensor], outputs=[combined_output_g], name='combined_model_g')

    combined_output_d = discriminator_model(inputs=[real_image_tensor, encoder_model(real_image_tensor)])
    combined_model_d = models.Model(inputs=[real_image_tensor], outputs=[combined_output_d], name='combined_model_d')

    #
    # define earth mover distance (wasserstein loss)
    #

    def em_loss(y_coefficients, y_pred):
        return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

    #
    # construct computation graph for calculating the gradient penalty (improved wGAN) and training the discriminator
    #

    # sample a batch of noise (generator input)
    latent_noise = tf.placeholder(tf.float32, shape=(batch_size, rand_dim))

    # sample a batch of real images
    real_images = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, img_channels))

    # generate a batch of images with the current generator
    generated_images = generator_model(latent_noise)

    generated_latent_noise = encoder_model(real_images)

    # calculate `x_hat`
    epsilon = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, 1))
    x_hat = epsilon * real_images + (1.0 - epsilon) * generated_images

    # calculate `z_hat`
    _epsilon = tf.placeholder(tf.float32, shape=(batch_size, rand_dim))
    z_hat = _epsilon * latent_noise + (1.0 - _epsilon) * generated_latent_noise

    # gradient penalty
    gradients = tf.gradients(discriminator_model(inputs=[x_hat, z_hat]), [x_hat, z_hat])
    _gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

    # calculate discriminator's loss
    _disc_loss = em_loss(tf.ones(batch_size), discriminator_model(inputs=[generated_images, latent_noise])) - \
        em_loss(tf.ones(batch_size), discriminator_model(inputs=[real_images, generated_latent_noise])) + \
        _gradient_penalty

    # update φ by taking an SGD step on mini-batch loss LD(φ)
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=.0001, beta1=0.5, beta2=0.9).minimize(
        _disc_loss, var_list=combined_model_d.trainable_weights)

    sess = K.get_session()

    #
    # compile models
    #

    adam = optimizers.Adam(lr=.0001, beta_1=0.5, beta_2=0.9)

    combined_model_d.compile(optimizer=adam, loss=[em_loss])
    discriminator_model.trainable = False

    combined_model_g.compile(optimizer=adam, loss=[em_loss])

    print(generator_model.summary())
    print(encoder_model.summary())
    print(discriminator_model.summary())
    print(combined_model_d.summary())
    print(combined_model_g.summary())

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
        **flow_from_directory_params
    )

    def get_image_batch():
        # PIL raises an exception for some invalid images in data set
        try:
            img_batch = real_image_generator.next()
        except OSError:
            return get_image_batch()

        # keras generators may generate an incomplete batch for the last batch in an epoch of data
        if len(img_batch) != batch_size:
            img_batch = real_image_generator.next()

        assert img_batch.shape == (batch_size, img_height, img_width, img_channels), img_batch.shape
        return img_batch

    disc_loss = []
    combined_loss = []

    def train_discriminator_step():
        d_l, _ = sess.run([_disc_loss, disc_optimizer], feed_dict={
            latent_noise: np.random.normal(loc=0.0, scale=1.0, size=(batch_size, rand_dim)),
            real_images: get_image_batch(),
            epsilon: np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1, 1)),
            _epsilon: np.random.uniform(low=0.0, high=1.0, size=(batch_size, rand_dim))
        })

        return d_l

    if generator_model_path:
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        discriminator_model.load_weights(discriminator_model_path, by_name=True)
    if encoder_model_path:
        encoder_model.load_weights(encoder_model_path, by_name=True)
    else:
        print('pre-training the critic...')

        for i in range(critic_pre_train_steps):
            print('Step: {} of {} critic pre-training.'.format(i, critic_pre_train_steps))
            loss = train_discriminator_step()

        print('Last batch of critic pre-training disc_loss: {}.'.format(loss))
        discriminator_model.save(os.path.join(cache_dir, 'discriminator_model_pre_trained.h5'))

    for i in range(nb_steps):
        # hacky but tensorflow/CUDA failing sporadically...
        try:
            print('Step: {} of {}.'.format(i, nb_steps))

            # train the discriminator
            for _ in range(k_d):
                # when plotting loss we will have to take `k_d` and `k_g` into account so the two plots align
                loss = train_discriminator_step()
                disc_loss.append(loss)

            # train the generator
            for _ in range(k_g):
                z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, rand_dim))

                # update θ by taking an SGD step on mini-batch loss LG(θ)
                loss = combined_model_g.train_on_batch(z, [-np.ones(batch_size)])
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
        except Exception as e:
            print(e)
            continue


def main(data_dir, generator_model_path, discriminator_model_path, encoder_model_path):
    adversarial_training(data_dir, generator_model_path, discriminator_model_path, encoder_model_path)


if __name__ == '__main__':
    gen_model_path = sys.argv[2] if len(sys.argv) >= 3 else None
    disc_model_path = sys.argv[3] if len(sys.argv) >= 4 else None
    enc_model_path = sys.argv[4] if len(sys.argv) >= 5 else None

    main(sys.argv[1], gen_model_path, disc_model_path, enc_model_path)
