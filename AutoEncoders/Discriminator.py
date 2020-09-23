import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminiator_model = self._create_discriminator()

    def _create_discriminator(self):
        NotImplementedError()

    def generate_real_samples(self, n_samples):
        NotImplementedError()

    def generate_fake_samples(self, n_samples):
        NotImplementedError()

    def train_discriminator(self):
        NotImplementedError()
