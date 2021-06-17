import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from os import path
import os


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @property
    def latentDim(self):
        return self.decoder.layers[0].input_shape[0][1]

    def train(self, images: np.ndarray, epochs: int = 30, batch_size: int = 128):
        self.compile(optimizer=keras.optimizers.Adam())
        self.fit(images, epochs=epochs, batch_size=batch_size)
    
    def getLatent(self, image: np.ndarray):
        out = self.encoder.predict(np.expand_dims(image, 0))[0]
        return out[0] if len(out.shape) == 2 else out

    def predict(self, latent: np.ndarray):
        img = self.decoder.predict(np.expand_dims(latent, 0))[0]
        return img

    def save(self, dir: str = '.'):
        if not path.exists(dir):
            os.mkdir(dir)
        self.encoder.save(path.join(dir, 'encoder.tf'))
        self.decoder.save(path.join(dir, 'decoder.tf'))

    @classmethod
    def load(cls, dir: str = '.', **kwargs):
        try:
            encoder = keras.models.load_model(path.join(dir, 'encoder.tf'))
            decoder = keras.models.load_model(path.join(dir, 'decoder.tf'))
            return cls(encoder, decoder, **kwargs)
        except OSError:
            return None

    @classmethod
    def loadFromBasic(cls, fileName: str, imgSize: tuple[int, int, int], latentDim: int, **kwargs):
        vae = buildBasicModel(imgSize, latentDim)
        # print([layer.trainable_weights for layer in vae.encoder.layers])
        layers = [vae.encoder.get_layer('encoder1'), vae.encoder.get_layer('encoder2'), vae.decoder.get_layer('decoder1'), vae.decoder.get_layer('decoder2')]
        # print([a.shape for a in vae.encoder.get_weights()])
        # print([layer.weights[0].shape for layer in vae.encoder.layers[2:]])
        with open(fileName, 'r') as file:
            for layer,line in zip(layers, file.readlines()):
                numbers = np.array([float(v) for v in line.split()])
                size = layer.weights[0].shape
                numbers = numbers.reshape((size[1], size[0]+1))
                numbers = numbers.transpose()
                bias = numbers[0]
                weights = numbers[1:]
                layer.set_weights([weights, bias])
        return vae

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def buildModel(imgShape: tuple[int, int, int], latentDim: int):

    encoder_inputs = keras.Input(shape=imgShape)
    x = layers.Flatten()(encoder_inputs)
    # x = layers.Dense(3888, activation="sigmoid")(x)
    x = layers.Dense(128, activation="sigmoid")(x)

    z_mean = layers.Dense(latentDim, name="z_mean")(x)
    z_log_var = layers.Dense(latentDim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = keras.Input(shape=(latentDim,))
    x = layers.Dense(128, activation="sigmoid")(latent_inputs)
    x = layers.Dense(np.product(imgShape), activation="sigmoid")(x)
    decoder_outputs = layers.Reshape(imgShape)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return VAE(encoder, decoder)

def buildBasicModel(imgShape: tuple[int, int, int], latentDim: int):

    encoder_inputs = keras.Input(shape=imgShape)
    x = layers.Permute((2, 1, 3))(encoder_inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="sigmoid", name="encoder1")(x)
    x = layers.Dense(latentDim, activation="sigmoid", name="encoder2")(x)

    encoder = keras.Model(encoder_inputs, x, name="encoder")
    encoder.summary()

    latent_inputs = keras.Input(shape=(latentDim,))
    x = layers.Dense(128, activation="sigmoid", name="decoder1")(latent_inputs)
    x = layers.Dense(np.product(imgShape), activation="sigmoid", name="decoder2")(x)
    x = layers.Reshape(imgShape)(x)
    decoder_outputs = layers.Permute((2, 1, 3))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return VAE(encoder, decoder)