import tensorflow as tf


class AutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        assert encoder is not None
        assert decoder is not None
        self.encoder = encoder
        self.decoder = decoder
        
    def call(self, inputs, training=None):
        latent_space = self.encoder(inputs, training=training)
        reconstructed_images = self.decoder(latent_space, training=training)
        return reconstructed_images