import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd

from Discriminator import Discriminator
from Generator import Generator


class MYFIRSTGAN(tf.keras.Model):
    def __init__(self, disc_model:tf.keras.Sequential, gen_model:tf.keras.Sequential):
        super(MYFIRSTGAN, self).__init__()
        self.gan = self._create_model(disc_model , gen_model)

    def _create_model(self, disc_model , gen_model):
        disc_model.trainable = False
        model = tf.keras.Sequential([
            disc_model,
            gen_model
        ]
        )
        opt = Adam(lr= 0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train_gan(self):
        pass




