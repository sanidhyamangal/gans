import tensorflow as tf  # for deep learning based ops
from dataclasses import dataclass

from .utils import generate_and_save_images


@dataclass
class BaseGANTraining:
    generator_model: tf.keras.Model
    discriminator_model: tf.keras.Model
    generator_optimizer: tf.optimizers.Optimizer
    discriminator_optimizer: tf.optimizers.Optimizer
    dataset: tf.data.Dataset
    epochs: int = 100
    save_images: bool = True
    checkpoint_at: int = 10

    @tf.function
    def train_step(self):
        pass

    def train(self):
        pass


@dataclass
class BaseTraining:
    model: tf.keras.Model
    loss_function: tf.losses.Loss
    optimizer: bool = tf.optimizers.Optimizer
    epochs: int = 100
    checkpoint_at: int = 10

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            pass

    def train(self):
        pass
