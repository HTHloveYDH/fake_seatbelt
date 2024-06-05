import tensorflow as tf


class AutoEncoderFakeModel(tf.keras.Model):
    def __init__(self, auto_encoder_1, auto_encoder_2, offset=0.0, scale=255.0, loss_scale=1.0, \
                 loss_fn_type='MSE', reduction='auto'):
        super(AutoEncoderFakeModel, self).__init__()
        self.auto_encoder_1 = auto_encoder_1  # this part is for src 
        self.auto_encoder_2 = auto_encoder_2  # this part is for dst
        self.loss_scale = loss_scale
        self.offset = offset
        self.scale = scale
        self.reduction = {
            'auto': tf.keras.losses.Reduction.AUTO, 
            'sum': tf.keras.losses.Reduction.SUM
        }[reduction]
        self.loss_fn = {
            'MAE': tf.keras.losses.MeanAbsoluteError(
                reduction=self.reduction, name='mean_absolute_error'
            ), 
            'MSE': tf.keras.losses.MeanSquaredError(
                reduction=self.reduction, name='mean_squared_error'
            )
        }[loss_fn_type]
    
    def call(self, inputs, training=None):
        data_1, data_2 = inputs
        reconstructed_images_1 = self.auto_encoder_1(data_1, training=training)
        reconstructed_images_2 = self.auto_encoder_2(data_2, training=training)
        return reconstructed_images_1, reconstructed_images_2
    
    def fake(self, inputs):
        assert inputs.ndim == 4 and inputs.shape[0] == 1
        latent_space = self.auto_encoder_1.encoder(inputs)
        fake_image = self.auto_encoder_2.decoder(latent_space)
        # fake_image = self.auto_encoder_1.decoder(latent_space)
        return (fake_image - self.offset) * self.scale
    
    def fake2(self, inputs):
        assert inputs.ndim == 4 and inputs.shape[0] == 1
        latent_space = self.auto_encoder_2.encoder(inputs)
        fake_image = self.auto_encoder_1.decoder(latent_space)
        return (fake_image - self.offset) * self.scale

    def train_step(self, data):
        data_1, data_2 = data
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass of student
            # [Note] do not forget 'training=True'
            # reconstructed_images_1, reconstructed_images_2 = self(data, training=True)
            reconstructed_images_1 = self.auto_encoder_1(data_1, training=True)
            reconstructed_images_2 = self.auto_encoder_2(data_2, training=True)
            loss_1 = self.loss_fn(data_1, reconstructed_images_1)
            loss_2 = self.loss_scale * self.loss_fn(data_2, reconstructed_images_2)
        # Compute gradients
        auto_encoder_1_trainable_vars = self.auto_encoder_1.trainable_variables  # extract trainable parameters of student model 
        auto_encoder_1_gradients = tape.gradient(loss_1, auto_encoder_1_trainable_vars)  # calculate gradients on each trainable parameter
        # Update weights
        self.optimizer.apply_gradients(zip(auto_encoder_1_gradients, auto_encoder_1_trainable_vars))  # BP algorithm
        # Compute gradients
        auto_encoder_2_trainable_vars = self.auto_encoder_2.trainable_variables  # extract trainable parameters of student model 
        auto_encoder_2_gradients = tape.gradient(loss_2, auto_encoder_2_trainable_vars)  # calculate gradients on each trainable parameter
        # Update weights
        self.optimizer.apply_gradients(zip(auto_encoder_2_gradients, auto_encoder_2_trainable_vars))  # BP algorithm
        # record performance 
        results = {}
        results.update({'loss': loss_1 + self.loss_scale * loss_2})
        return results

    def test_step(self, data):
        data_1, data_2 = data
        # reconstructed_images_1, reconstructed_images_2 = self(data, training=False)
        reconstructed_images_1 = self.auto_encoder_1(data_1, training=False)
        reconstructed_images_2 = self.auto_encoder_2(data_2, training=False)
        loss_1 = self.loss_fn(data_1, reconstructed_images_1)
        loss_2 = self.loss_fn(data_2, reconstructed_images_2)
        # record performance 
        results = {}
        results.update({'loss': loss_1 + self.loss_scale * loss_2})
        return results