"""
author: Sanidhya Mangal
github: sanidhyamangal
"""

import os  # for os related ops
from pathlib import Path  # for path matching
from typing import Optional, Tuple  # for typings

import tensorflow as tf  # for deep learning based ops


class FileDataLoader:
    """
    A data loader class for loading all the images from a file
    """
    # base file decoder
    FILE_DECODER = {'jpg': tf.image.decode_jpeg, 'png': tf.image.decode_png}

    def __init__(self,
                 path_to_images: str,
                 image_extension: str,
                 image_dims: Tuple[int],
                 image_channels: int = 1,
                 **kwargs):
        self.image_channels = image_channels
        self.image_extension = image_extension
        if len(image_dims) != 2:
            raise AttributeError("image dims can only be 2 dim integers")

        self.image_dims = image_dims
        _image_path = Path(path_to_images)
        self.image_list = [
            str(image)
            for image in _image_path.glob(f"*.{self.image_extension}")
        ]

    def process_images(self, file_path: str, *args, **kwargs):
        raw_image = tf.io.read_file(file_path)
        image = self.FILE_DECODER[self.image_extension](raw_image)
        image = tf.image.resize(image, size=(64, 64))

        return (image - 127.0) / 127.0

    def create_dataset(self,
                       batch_size: int,
                       shuffle: bool = True,
                       autotune: Optional = None,
                       **kwargs):

        cache = kwargs.pop('cache', False)
        prefetch = kwargs.pop('prefetch', False)
        ds = tf.data.Dataset.from_tensor_slices(self.image_list).map(
            self.process_images, num_parallel_calls=autotune)

        # shuffle the dataset if present
        if shuffle:
            ds = ds.shuffle(len(self.image_list))

        # create a batch of dataset
        ds = ds.batch(batch_size)

        # check if cache is enabled or not
        if cache:
            ds = ds.cache()

        # check if prefetch is specified or not
        if prefetch:
            ds = ds.prefetch(prefetch)

        return ds
