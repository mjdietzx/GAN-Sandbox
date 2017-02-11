"""
Module to plot generated images and save the plots to disc as we train our GAN.

"""

# TODO: share this module w/ SimGAN repo

import os

import matplotlib
import numpy as np

matplotlib.use('Agg')  # b/c matplotlib is such a great piece of software ;) - needed to work on ubuntu
from matplotlib import pyplot as plt


def plot_batch(generated_image_batch, real_image_batch, figure_path):
    """
    Generate a plot of `batch_size` real and generated images and save it to disc.

    :param generated_image_batch: Batch of generated images.
    :param real_image_batch: Batch of real images.
    :param figure_path: Full path of file name the plot will be saved as.
    """
    batch_size = generated_image_batch.shape[0]
    image_batch = np.concatenate((generated_image_batch, real_image_batch))

    nb_rows = batch_size // 10 + 1
    nb_columns = 10 * 2

    _, ax = plt.subplots(nb_rows, nb_columns, sharex=True, sharey=True)

    for i in range(nb_rows):
        for j in range(0, nb_columns, 2):
            try:
                # pre-processing function, applications.xception.preprocess_input => [0.0, 1.0]
                ax[i][j].imshow((image_batch[i * nb_columns + j] / 2.0 + 0.5))
                ax[i][j + 1].imshow((image_batch[i * nb_columns + j + batch_size] / 2.0 + 0.5))
            except IndexError:
                pass
            ax[i][j].set_axis_off()
    plt.savefig(os.path.join(figure_path), dpi=600)
    plt.close()
