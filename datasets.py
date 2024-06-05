import os
import json

import tensorflow as tf


def load_image(filename, channels:int):
    image_byte_string = tf.io.read_file(filename)  # for jpg, jpeg, bmp, png, gif format
    image = tf.io.decode_image(image_byte_string, channels=channels, expand_animations=False)
    # image = tf.io.decode_jpeg(image_byte_string, channels=3)  # only for jpg, jpeg format
    return image


def load_images(filenames):
    # filenames has been batched
    images = tf.vectorized_map(load_image, filenames, fallback_to_while_loop=True, warn=True)
    return images


def resize_image(images, initial_img_width:int, initial_img_height:int):
    images = tf.image.resize(images, [initial_img_height, initial_img_width])
    return images


def rescale_image(images, scale:float, offset:float):
    images = images / scale + offset
    return images


def configure_dataset(dataset, batch_size:int, shuffle_buffer_size:int, is_repeat:bool, channels:int, \
                      initial_img_width:int, initial_img_height:int, scale:float, offset:float):
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)  # default to reshuffle every repeat
    if is_repeat:
        dataset = dataset.repeat()  # default to infinite
    dataset = dataset.map(lambda x: load_image(x, channels), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: resize_image(x,  initial_img_width, initial_img_height), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x: rescale_image(x, scale, offset), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_dataset(annotation_filename_1:str, annotation_filename_2:str, batch_size:int, shuffle_buffer_size:int, \
                 is_repeat:bool, channels:int, initial_img_width:int, initial_img_height:int, scale:float, \
                 offset:float):
    if not os.path.exists(annotation_filename_1) or not os.path.exists(annotation_filename_2):
        return None, 0
    with open(annotation_filename_1, 'r') as f:
        annotations_1 = json.load(f)['annotations']
    with open(annotation_filename_2, 'r') as f:
        annotations_2 = json.load(f)['annotations']
    assert len(annotations_1) == len(annotations_2)
    dataset_length = len(annotations_1)
    filenames_1 = []
    for filename in annotations_1:
        filenames_1.append(filename)
    filenames_1 = tf.cast(filenames_1, tf.string)
    dataset_1 = tf.data.Dataset.from_tensor_slices(filenames_1)
    dataset_1 = configure_dataset(
        dataset_1, batch_size, shuffle_buffer_size, is_repeat, channels, initial_img_width, 
        initial_img_height, scale, offset
    )
    filenames_2 = []
    for filename in annotations_2:
        filenames_2.append(filename)
    filenames_2 = tf.cast(filenames_2, tf.string)
    dataset_2 = tf.data.Dataset.from_tensor_slices(filenames_2)
    dataset_2 = configure_dataset(
        dataset_2, batch_size, shuffle_buffer_size, is_repeat, channels, initial_img_width, 
        initial_img_height, scale, offset
    )
    if tf.__version__ in ['2.14.0']:
        dataset = tf.data.Dataset.zip(dataset_1, dataset_2)
    elif tf.__version__ in ['2.10.0', '2.10.1', '2.12.0']:
        dataset = tf.data.Dataset.zip((dataset_1, dataset_2))
    else:
        raise ValueError(f'tensorflow version: {tf.__version__} is not supported')
    return dataset, dataset_length