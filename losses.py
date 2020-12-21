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
