"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import os  # for os related ops
from datetime import datetime  # for date related ops
from typing import Optional  # for typings

import tensorflow as tf  # for deep learning based ops

from .losses import WasserstienLossMixin  # load loss mixins
from .utils import \
    generate_and_save_images  # for saving and generation of image


class BaseGANTrainer(tf.Module):
    discriminator_loss = None
    generator_loss = None

    def __init__(self,
                 generator_model: tf.keras.models.Model,
                 discriminator_model: tf.keras.Model,
                 generator_optimizer: tf.optimizers.Optimizer,
                 discriminator_optimizer: tf.optimizers.Optimizer,
                 epochs: int = 100,
                 save_images: bool = True,
                 save_checkpoint_at: int = 0,
                 checkpoint_dir: Optional[str] = None,
                 *args,
                 **kwargs):
        """
        A class for performing training operations on GANs
        """

        # training hyper parameter tuner
        self.generator = generator_model
        self.discriminator = discriminator_model
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.save_images = save_images
        self.epochs = epochs
        self.save_checkpoint_at = save_checkpoint_at
        self.multi_channel = True if self.generator.model.output_shape[
            -1] > 1 else False

        # check if checkpoint dir and save checkpoint at configured or not
        if self.save_checkpoint_at and not checkpoint_dir:
            raise AttributeError(
                "%s should include checkpoint_dir if save_checkpoint is set",
                self.__class__.__name__)

        # create instance for checkpoint based on checkpoint_dirß
        if self.save_checkpoint_at and checkpoint_dir:
            self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            self.checkpoint = tf.train.Checkpoint(
                generator=self.generator,
                discriminator=self.discriminator,
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer)

    def get_generator_loss(self, logits):
        assert self.generator_loss is not None, (
            "%s should include generator_loss attribute or override `get_generator_loss method`"
            % self.__class__.__name__)

        return self.generator_loss(logits)

    def get_discriminator_loss(self, real_logits, fake_logits):
        assert self.generator_loss is not None, (
            "%s should include discriminator_loss attribute or override `get_discriminator_loss()` method"
            % self.__class__.__name__)

        return self.discriminator_loss(real_logits, fake_logits)

    def train(self, dataset: tf.data.Dataset, batch_size: int, noise_dim: int):
        # seed for constant image gen ops
        self.seed = tf.random.normal([16, noise_dim])
        for epoch in range(self.epochs):

            start_time = datetime.now()

            for images in dataset:
                noise = tf.random.normal([batch_size, noise_dim])
                self.train_step(images, noise)

            # call for saving and generation of images
            if self.save_images:
                generate_and_save_images(
                    self.generator,
                    self.seed,
                    image_name="image_at_{}.png".format(epoch),
                    multi_channel=self.multi_channel)

            # saving checkpoints if specified
            if self.save_checkpoint_at != 0:
                if (epoch + 1) % self.save_checkpoint_at == 0:
                    self.checkpoint(self.checkpoint_prefix)

            print(f"Time for epoch: {epoch+1} is {datetime.now()- start_time}")

    @tf.function
    def train_step(self, images, noise):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # call for model
            generated_output = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_output, training=True)

            # get loss
            gen_loss = self.get_generator_loss(fake_output)
            disc_loss = self.get_discriminator_loss(real_output, fake_output)

        # get gradients wrt loss
        generator_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        # apply gradients
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients,
                self.discriminator.trainable_variables))


class WasserstienGANTrainer(WasserstienLossMixin, BaseGANTrainer):
    """
    A Class for performing wasserstien gan ops
    """
    generator_loss = True
