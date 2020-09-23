import numpy as np
import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.generator_model = self.create_generator_model()

    def _create_generator_model(self, latent_dim):
        pass

    def train_generator(self):
        pass

    def generate_fake_sample(self, latent_dim, n_samples):
        pass

        def _generator_noise_input(latent_dim, n_samples):
            pass
