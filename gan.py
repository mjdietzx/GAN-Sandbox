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

img_height = 56
img_width = 56
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

cardinality = 16


def add_common_layers(y):
    # y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    return y


def grouped_convolution(y, nb_channels, _strides, _transposed=False):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        if _strides != (1, 1) and _transposed:
            return layers.Conv2DTranspose(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        else:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)

    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        if _strides != (1, 1) and _transposed:
            groups.append(layers.Conv2DTranspose(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
        else:
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = layers.concatenate(groups)

    return y


def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False, _transposed=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
      * If up-sampled in case of `_transposed` == True, the width of the blocks is divided by a factor of 2.
    """
    shortcut = y

    # we modify the residual building block as a bottleneck design to make the network more economical
    y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = add_common_layers(y)

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = grouped_convolution(y, nb_channels_in, _strides=_strides, _transposed=_transposed)
    y = add_common_layers(y)

    y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    # y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        if _strides != (1, 1) and _transposed:
            shortcut = layers.Conv2DTranspose(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        else:
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)

        # shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = layers.LeakyReLU()(y)

    return y


def stack_blocks(x, transposed=False):
    # conv2
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 64, 256, _strides=strides, _transposed=transposed)

    # conv3
    for i in range(2):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 512, _strides=strides, _transposed=transposed)

    # conv4
    """for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 1024, _strides=strides, _transposed=transposed)

    # conv5
    for i in range(2):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 2048, _strides=strides, _transposed=transposed)"""

    return x


def generator_network(x):
    x = layers.Dense(64 * 7 * 7)(x)
    x = add_common_layers(x)

    x = layers.Reshape((7, 7, 64))(x)
    x = stack_blocks(x, transposed=True)

    # conv5 (conv1 disc)
    # number of feature maps => number of image channels
    return layers.Conv2DTranspose(img_channels, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='tanh')(x)


def discriminator_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    """
    # conv1
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    x = stack_blocks(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1)(x)

    return x


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
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')

    #
    # define earth mover distance (wasserstein loss)
    #

    def em_loss(y_coefficients, y_pred):
        return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

    #
    # construct computation graph for calculating the gradient penalty (improved wGAN) and training the discriminator
    #

    # sample a batch of noise (generator input)
    _z = tf.placeholder(tf.float32, shape=(batch_size, rand_dim))

    # sample a batch of real images
    _x = tf.placeholder(tf.float32, shape=(batch_size, img_height, img_width, img_channels))

    # generate a batch of images with the current generator
    _g_z = generator_model(_z)

    # calculate `x_hat`
    epsilon = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, 1))
    x_hat = epsilon * _x + (1.0 - epsilon) * _g_z

    # gradient penalty
    gradients = tf.gradients(discriminator_model(x_hat), [x_hat])
    _gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

    # calculate discriminator's loss
    _disc_loss = em_loss(tf.ones(batch_size), discriminator_model(_g_z)) - \
        em_loss(tf.ones(batch_size), discriminator_model(_x)) + \
        _gradient_penalty

    # update φ by taking an SGD step on mini-batch loss LD(φ)
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=.0001, beta1=0.5, beta2=0.9).minimize(
        _disc_loss, var_list=discriminator_model.trainable_weights)

    sess = K.get_session()

    #
    # compile models
    #

    adam = optimizers.Adam(lr=.0001, beta_1=0.5, beta_2=0.9)

    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss=[em_loss])

    print(generator_model.summary())
    print(discriminator_model.summary())
    print(combined_model.summary())

    # assert (len(discriminator_model.layers) - 4) / 6 == N

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
        img_batch = real_image_generator.next()

        # keras generators may generate an incomplete batch for the last batch in an epoch of data
        if len(img_batch) != batch_size:
            img_batch = real_image_generator.next()

        assert img_batch.shape == (batch_size, img_height, img_width, img_channels), img_batch.shape
        return img_batch

    disc_loss = []
    combined_loss = []

    def train_discriminator_step():
        d_l, _ = sess.run([_disc_loss, disc_optimizer], feed_dict={
            _z: np.random.normal(loc=0.0, scale=1.0, size=(batch_size, rand_dim)),
            _x: get_image_batch(),
            epsilon: np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1, 1, 1))
        })

        return d_l

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
        except Exception as e:
            print(e)
            continue


def main(data_dir, generator_model_path, discriminator_model_path):
    adversarial_training(data_dir, generator_model_path, discriminator_model_path)


if __name__ == '__main__':
    gen_model_path = sys.argv[2] if len(sys.argv) >= 3 else None
    disc_model_path = sys.argv[3] if len(sys.argv) >= 4 else None

    main(sys.argv[1], gen_model_path, disc_model_path)
