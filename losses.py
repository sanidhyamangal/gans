"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import tensorflow as tf  # for deep learning based ops


class WasserstienLossMixin:
    """
    A mixin class for overriding loss for generator loss and discriminator loss
    """
    def get_generator_loss(self, logits) -> tf.Tensor:
        """
        A function to return generator loss for wasserstein gan architecture
        :param logits: output of the generator gan
        :return: reduced mean of input logits
        """
        return -tf.reduce_mean(logits)

    def get_discriminator_loss(self, real_logits, fake_logits) -> tf.Tensor:
        """
        Function to return discriminator loss for wassesrstein gan architecture
        :param real_logits: output of the discriminator for the real images
        :param fake_logits: output of the discriminator from the generated images
        :return: difference of reduced mean of real logits from fake logits
        """
        return tf.reduce_sum(fake_logits) - tf.reduce_sum(real_logits)


class BaseCriterionLossMixin:
    """
    Base class for creation of criterion based losss such as binary cross entropy or mean squared or any other
    """
    loss_criterion = tf.losses.Loss  # base loss criterion function

    def get_generator_loss(self, fake_output) -> tf.Tensor:
        """
        A function to compute generator loss for any given loss criterion
        :param fake_output: output of a discriminator which data feed as generated image
        :return: loss based on loss criterion
        """

        return self.loss_criterion(tf.ones_like(fake_output), fake_output)

    def get_discriminator_loss(self, real_output, fake_output) -> tf.Tensor:
        """
        A function for computing discriminator loss for any given loss criterion
        :param real_output: output of discriminator from real data set
        :param fake_output: output of discriminator from fake data generated by generator
        :return: loss based on loss criterion
        """

        real_loss = self.loss_criterion(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_criterion(tf.zeros_like(fake_output),
                                        fake_output)

        return real_loss + fake_loss


class DCGANLossMixin(BaseCriterionLossMixin):
    """
    A mixin class for overriding loss methods for generators and discriminators for deep convolution bases networks.
    """
    loss_criterion = tf.losses.BinaryCrossentropy(from_logits=True)


class LSGANLossMixin(BaseCriterionLossMixin):
    """
    A mixin class for overriding generator loss and discriminator loss for mean squared loss criterion
    """
    loss_criterion = tf.losses.MeanSquaredError()
