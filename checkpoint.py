"""
author: Sanidhya Mangal
github: sanidhyamangal
"""

import os  # for os related ops

import tensorflow as tf  # for deep learning based ops


class BaseCheckpointSaverMixin:
    """
    A mixin class for saving checkpoints at a random step
    """
    def save_checkpoint(self,
                        checkpoint_dir: str = "./training_checkpoint") -> None:

        assert self.generator is None, (
            "%s must have generator for creating a checkpoint" %
            self.__class__.__name__)

        assert self.discriminator is None, (
            "%s must have discriminator object for creating a checkpoint" %
            self.__class__.__name__)

        assert self.discriminator_optimizer is None, (
            "%s must have discriminator_optimizer object for creating a checkpoint"
            % self.__class__.__name__)

        assert self.generator_optimizer is None, (
            "%s must have generator_optimizer object for creating a checkpoint"
            % self.__class__.__name__)
        
        checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt")
        checkpoint = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator, discriminator_optimizer=self.discriminator_optimizer, generator_optimizer=self.generator_optimizer)

        checkpoint.save(checkpoint_prefix)
