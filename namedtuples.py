"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
from datetime import datetime  # for date related ops

import tensorflow as tf  # for deep learning based ops

from losses import (  # losses for wasserstien models
    wasserstien_discriminator_loss, wasserstien_generator_loss)
from utils import generate_and_save_images  # for saving and generation of image


class BaseGANTrainer:
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
        self.seed = tf.random.normal([batch_size, noise_dim])
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
                        self.multi_channel)

            # TODO: implement the stuff for saving checkpoints mechanism
            if self.save_checkpoint_at != 0:
                if (epoch + 1) % self.save_checkpoint_at == 0:
                    pass

            print(f"Time for epoch: {epoch+1} is {datetime.now()- start_time}")

    @tf.function
    def train_step(self, images, noise):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # call for model
            generated_output = self.generator(noise)
            real_output = self.discriminator(images)
            fake_output = self.discriminator(generated_output)

            # get loss
            gen_loss = self.get_generator_loss(generated_output)
            disc_loss = self.get_discriminator_loss(real_output, fake_output)

        # get gradients wrt loss
        generator_gradients = gen_tape.gradient(
            gen_loss, self.genertor.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        # apply gradients
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients,
                self.discriminator.trainable_variables))


class WasserstienGANTrainer(BaseGANTrainer):
    generator_loss = wasserstien_generator_loss
    discriminator_loss = wasserstien_discriminator_loss
