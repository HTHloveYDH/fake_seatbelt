import json
import random

import tensorflow as tf

from callbacks import SaveCurrentAutoEncoder, AutoEncoderModelCheckpoint


def generate_annotation_file(src_annotation_file:str, num:int, usage:str, index:int):
    json_data = {'annotations': []}
    with open(src_annotation_file,'r') as f:
        src_content = json.load(f)
    random.shuffle(src_content['annotations'])
    for img_info in src_content['annotations'][:num]:
        json_data['annotations'].append(img_info['filename'])
    with open(f'./anno_{usage}_{index}.json','w') as f:
        json.dump(json_data, f)


def get_optimizer(optimizer_type:str):
    if optimizer_type == 'SGD':
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.0)
    elif optimizer_type == 'Adam':
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, 
            name='Adam'
        )
    else:
        raise ValueError(f'optimizer_type: {optimizer_type} is not supported')
    return optimizer


def set_callbacks(flag):
    callbacks = []
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta=0.001, mode='min')
    callbacks.append(early_stopping)
    if flag:
        checkpointer = AutoEncoderModelCheckpoint(
            -1, './saved_model/model', save_weights_only=False, monitor="val_loss", save_best_only=True, 
            mode="min"
        )
        callbacks.append(checkpointer)
    else:
        # checkpointer = SaveCurrentAutoEncoder('./saved_model/model')
        checkpointer = AutoEncoderModelCheckpoint(
            -1, './saved_model/model', save_weights_only=False, monitor="loss", save_best_only=True, 
            mode="min"
        )
        callbacks.append(checkpointer)
    return callbacks