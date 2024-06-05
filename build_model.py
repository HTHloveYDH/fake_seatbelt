import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from AutoEncoder import AutoEncoder
from AutoEncoderFakeModel import AutoEncoderFakeModel


def build_encoder(input_shape:tuple, filters:int, latent_dim:int, channel_dim=-1):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    # loop over the filters
    for filter in filters:
        # Build network with Convolutional with RELU and BatchNormalization
        x = tf.keras.layers.Conv2D(filter, (3, 3), strides=2, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_dim)(x)
    # flatten the network and then construct the latent vector
    volume_size = K.int_shape(x)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(latent_dim, kernel_regularizer=tf.keras.regularizers.L2(l2=0.1))(x)
    # build the encoder model
    encoder = tf.keras.Model(inputs, latent, name="encoder")
    return encoder, volume_size


def build_decoder(input_shape:tuple, filters:int, latent_dim:int, volume_size, channel_dim=-1):
    latent = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(np.prod(volume_size[1:]), kernel_regularizer=tf.keras.regularizers.L2(l2=0.1))(latent)
    x = tf.keras.layers.Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
    # We will loop over the filters again but in the reverse order
    for filter in filters[::-1]:
        # In the decoder, we will apply a CONV_TRANSPOSE with RELU and BatchNormalization operation
        x = tf.keras.layers.Conv2DTranspose(filter, (3, 3), strides=2, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization(axis=channel_dim)(x)
    # Now, we want to recover the original depth of the image. For this, we apply a single CONV_TRANSPOSE layer
    x = tf.keras.layers.Conv2DTranspose(input_shape[channel_dim], (3, 3), padding="same")(x)
    outputs = tf.keras.layers.Activation("sigmoid")(x)
    # Now build the decoder model
    decoder = tf.keras.Model(latent, outputs, name="decoder")
    return decoder


def build_auto_encoder(input_shape:tuple, filters:int, latent_dim:int, channel_dim=-1):
    encoder, volume_size = build_encoder(input_shape, filters, latent_dim, channel_dim)
    decoder = build_decoder(input_shape, filters, latent_dim, volume_size, channel_dim)
    return AutoEncoder(encoder, decoder), volume_size


def build_auto_encoder_fake_model(input_shape:tuple, filters:int, latent_dim:int, loss_scale:float, \
                                  offset:float, scale:float, loss_fn_type:str, channel_dim=-1):
    # share encoder
    auto_encoder_1, volume_size = build_auto_encoder(input_shape, filters, latent_dim, channel_dim)
    decoder_2 = build_decoder(input_shape, filters, latent_dim, volume_size, channel_dim)
    auto_encoder_2 = AutoEncoder(auto_encoder_1.encoder, decoder_2)
    return AutoEncoderFakeModel(auto_encoder_1, auto_encoder_2, offset, scale, loss_scale, loss_fn_type)


def load_auto_encoder_fake_model(load_autocoder_1_path:str, load_autocoder_2_path:str, offset:float, scale:float):
    custom_objects = {
        'AutoEncoder': AutoEncoder, 
        'AutoEncoderFakeModel': AutoEncoderFakeModel
    }
    auto_encoder_1 = tf.keras.models.load_model(
        load_autocoder_1_path, custom_objects=custom_objects, compile=False
    )
    auto_encoder_2 = tf.keras.models.load_model(
        load_autocoder_2_path, custom_objects=custom_objects, compile=False
    )
    return AutoEncoderFakeModel(auto_encoder_1, auto_encoder_2, offset, scale)
