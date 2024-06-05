import os
import logging
from datetime import datetime

import tensorflow as tf
try:
    from keras.utils import io_utils
    from keras.utils import tf_utils
except:
    from tensorflow.python.keras.utils import io_utils
    from tensorflow.python.keras.utils import tf_utils


class AutoEncoderModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(
        self, 
        threshold,
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
        options=None,
        initial_value_threshold=None,
        **kwargs
    ):
        super(AutoEncoderModelCheckpoint, self).__init__(
            filepath,
            monitor,
            verbose,
            save_best_only,
            save_weights_only,
            mode,
            save_freq,
            options,
            initial_value_threshold,
            **kwargs
        )
        self.threshold=threshold
        if threshold > 0:
            self.continous_not_improved_epochs = 0
            self.learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                self.model.optimizer.lr, decay_steps=1, decay_rate=0.98, staircase=True, name=None
            )
        else:
            self.continous_not_improved_epochs = None
            self.learning_rate_scheduler = None
    
    def _save_model(self, epoch, batch, logs):
        """Saves the model.
        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                   is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning(
                            'Can save best model only with %s available, '
                            'skipping.', 
                            self.monitor
                        )
                    else:
                        if self.monitor_op(current, self.best):
                            if self.threshold > 0:
                                self.continous_not_improved_epochs = 0
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: {self.monitor} improved '
                                    f'from {self.best:.5f} to {current:.5f}, '
                                    f'saving model to {filepath}'
                                )
                            self.best = current
                            if self.save_weights_only:
                                if hasattr(self.model, 'auto_encoder_1') and hasattr(self.model, 'auto_encoder_2'):
                                    self.model.auto_encoder_1.save_weights(os.path.join(filepath, 'auto_encoder_1'), overwrite=True, options=self._options)
                                    self.model.auto_encoder_2.save_weights(os.path.join(filepath, 'auto_encoder_2'), overwrite=True, options=self._options)
                                else:
                                    self.model.save_weights(filepath, overwrite=True, options=self._options)
                            else:
                                if hasattr(self.model, 'auto_encoder_1') and hasattr(self.model, 'auto_encoder_2'):
                                    self.model.auto_encoder_1.save(os.path.join(filepath, 'auto_encoder_1'), overwrite=True, options=self._options)
                                    self.model.auto_encoder_2.save(os.path.join(filepath, 'auto_encoder_2'), overwrite=True, options=self._options)
                                else:  
                                    self.model.save(filepath, overwrite=True, options=self._options)
                        else:
                            if self.threshold > 0:
                                self.continous_not_improved_epochs += 1
                                if self.continous_not_improved_epochs >= self.threshold:
                                    self.model.optimizer.lr = self.learning_rate_scheduler(self.model.optimizer.lr)
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: '
                                    f'{self.monitor} did not improve from {self.best:.5f}'
                                )
                else:
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f'\nEpoch {epoch + 1}: saving model to {filepath}')
                    if self.save_weights_only:
                        if hasattr(self.model, 'auto_encoder_1') and hasattr(self.model, 'auto_encoder_2'):
                            self.model.auto_encoder_1.save_weights(os.path.join(filepath, 'auto_encoder_1'), overwrite=True, options=self._options)
                            self.model.auto_encoder_2.save_weights(os.path.join(filepath, 'auto_encoder_2'), overwrite=True, options=self._options)
                        else:
                            self.model.save_weights(filepath, overwrite=True, options=self._options)
                    else:
                        if hasattr(self.model, 'auto_encoder_1') and hasattr(self.model, 'auto_encoder_2'):
                            self.model.auto_encoder_1.save(os.path.join(filepath, 'auto_encoder_1'), overwrite=True, options=self._options)
                            self.model.auto_encoder_2.save(os.path.join(filepath, 'auto_encoder_2'), overwrite=True, options=self._options)
                        else:    
                            self.model.save(filepath, overwrite=True, options=self._options)
                self._maybe_remove_file()
            except IsADirectoryError as e:  # h5py 3.x
                raise IOError(
                          'Please specify a non-directory filepath for '
                          'ModelCheckpoint. Filepath used is an existing '
                          f'directory: {filepath}'
                      )
            except IOError as e:  # h5py 2.x
            # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError(
                              'Please specify a non-directory filepath for '
                              'ModelCheckpoint. Filepath used is an existing '
                              f'directory: f{filepath}'
                          )
                # Re-throw the error for any other causes.
                raise e
     
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.save_freq == 'epoch':
          self._save_model(epoch=epoch, batch=None, logs=logs)


class SaveCurrentAutoEncoder(tf.keras.callbacks.Callback):
    def __init__(self, save_model_dir:str):
        super(SaveCurrentAutoEncoder, self).__init__()
        self.save_model_dir = save_model_dir
        self.save_model_dir_1 = os.path.join(save_model_dir, 'auto_encoder_1')
        self.save_model_dir_2 = os.path.join(save_model_dir, 'auto_encoder_2')
        if not os.path.exists(os.path.join(self.save_model_dir_1, 'history')):
            os.mkdir(os.path.join(self.save_model_dir_1, 'history'))
        if not os.path.exists(os.path.join(self.save_model_dir_2, 'history')):
            os.mkdir(os.path.join(self.save_model_dir_2, 'history'))

    def on_epoch_end(self, epoch, logs=None):
        now = datetime.now() # current date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        # just save self.model.kernel_model, not entire self.model (for keras subclass model)
        if hasattr(self.model, 'auto_encoder_1') and hasattr(self.model, 'auto_encoder_2'):
            save_model_path_1 = os.path.join(self.save_model_dir_1, 'history', f'Epoch_{epoch}_{date_time}')
            save_model_path_2 = os.path.join(self.save_model_dir_2, 'history', f'Epoch_{epoch}_{date_time}')
            os.mkdir(save_model_path_1, save_model_path_2)
            tf.keras.models.save_model(self.model.auto_encoder_1, save_model_path_1, overwrite=True)
            tf.keras.models.save_model(self.model.auto_encoder_2, save_model_path_2, overwrite=True)
        # just save self.model.model, not entire self.model (for keras fuctional / sequential model)
        else:
            save_model_path = os.path.join(self.save_model_dir, 'history', f'Epoch_{epoch}_{date_time}')
            os.mkdir(save_model_path)
            tf.keras.models.save_model(self.model, save_model_path, overwrite=True)
        print(f'saving current model to {str(save_model_path)}')