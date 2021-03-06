"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
from functools import partial  # for partial functions
from typing import List, Tuple

import tensorflow as tf  # for deep learning based ops

from .residual.blocks import ResidualBasicBlock, ResidualBottleNeckBlock


class ConvolutionalGeneratorModel(tf.keras.models.Model):
    def __init__(self,
                 channel_dim: int = 1,
                 filters: List[int] = [],
                 strides: Tuple[int, int] = (2, 2),
                 kernel_size: Tuple[int, int] = (5, 5),
                 padding: str = "same",
                 shape: Tuple[int, int] = (8, 8),
                 input_shape: int = 100,
                 **kwargs):
        """
        A Generator model for performing the ConvolutionalGeneration models ops
        :param channel_dim: Dimensions of images which needs to be generated
        :param filters: list of filters to use it to iterate in generation of conv2d
        transpose layer
        :param strides: strides values for conv2d transpose layers, default 5x5
        :param kernel_size: kernel size for conv2d transpose layers, default 2x2
        :param padding: padding values for the conv2d transpose layer, default same
        :param shape: input shape to begin with, default 8x8
        :param input_shape: input shape for the noise seed, default 100
        :param args:
        :param kwargs:
        """
        super(ConvolutionalGeneratorModel, self).__init__(**kwargs)
        Conv2DT = partial(tf.keras.layers.Conv2DTranspose,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          use_bias=False)
        model_layers = [
            tf.keras.layers.Dense((shape[0] * shape[1] * filters[0]),
                                  use_bias=False,
                                  input_shape=(input_shape, )),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((shape[0], shape[1], filters[0]))
        ]

        model_layers.extend([
            Conv2DT(filters=filters[1], strides=(1, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
        ])

        for _filter in filters[2:]:
            _conv_stack = [
                Conv2DT(filters=_filter),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU()
            ]
            model_layers.extend(_conv_stack)

        model_layers.append(Conv2DT(filters=channel_dim,
                                    activation=tf.nn.tanh))
        self.model = tf.keras.models.Sequential(model_layers)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class ConvolutionalDiscriminativeModel(tf.keras.models.Model):
    def __init__(self,
                 filters: List[int] = [],
                 strides: Tuple[int, int] = (2, 2),
                 kernel_size: Tuple[int, int] = (5, 5),
                 padding: str = "same",
                 dropout_rate: float = 0.3,
                 **kwargs):
        """
        A Discriminator model for performing the ConvolutionalDiscriminative models ops
        :param filters: list of filters to use it to iterate in generation of conv2d
        transpose layer
        :param strides: strides values for conv2d transpose layers, default 5x5
        :param kernel_size: kernel size for conv2d transpose layers, default 2x2
        :param padding: padding values for the conv2d transpose layer, default same
        :param dropout_rate: rate of dropout which needs to be used in dropout layer
        :param args:
        :param kwargs:
        """
        super(ConvolutionalDiscriminativeModel, self).__init__(**kwargs)
        Conv2D = partial(tf.keras.layers.Conv2D,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding)
        model_layers = []

        for _filter in filters:
            _conv_stack = [
                Conv2D(filters=_filter),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(dropout_rate)
            ]
            model_layers.extend(_conv_stack)

        model_layers.extend(
            [tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(1)])
        self.model = tf.keras.models.Sequential(model_layers)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)


class ResidualConvolutionalDiscriminative(tf.keras.models.Model):
    """
    A Model class for implementing discriminator using ResnetArchitecture
    """
    RESIDUAL_TYPE = {
        "basic": ResidualBasicBlock,
        "bottleneck": ResidualBottleNeckBlock
    }

    def __init__(self,
                 *args,
                 filters: List[int] = [],
                 strides: Tuple[int, int] = (2, 2),
                 kernel_size: Tuple[int, int] = (5, 5),
                 padding: str = "same",
                 block_type: str = "basic",
                 dropout_rate: float = 0.3,
                 **kwargs):

        super(ResidualConvolutionalDiscriminative,
              self).__init__(*args, **kwargs)

        if block_type not in ["basic", "bottle_neck"]:
            raise ValueError("Invalid block_type %s", block_type)

        # create a residual block based on block type

        model_layers = [
            tf.keras.layers.Convolution2D(filters=64,
                                          kernel_size=(7, 7),
                                          padding="same"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(dropout_rate)
        ]

        for _filters in filters:
            _conv_stack = [
                self.RESIDUAL_TYPE[block_type](filters=_filters,
                                               kernel_size=kernel_size,
                                               padding=padding,
                                               strides=strides),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(dropout_rate)
            ]

            model_layers.extend(_conv_stack)

        model_layers.extend(
            [tf.keras.layers.Flatten(),
             tf.keras.layers.Dense(1)])

        self.model = tf.keras.models.Sequential(model_layers)

    def call(self, inputs, *args, **kwargs):
        training = kwargs.pop('training', True)
        return self.model(inputs, training=training)
