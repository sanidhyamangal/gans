# gans
This is a light weight framework on top of gans for tooling Generative Adversarial Networks(GANs) for TF2.0 supporting eager execution and keras tooling.

## Structure of GANs library
- [models](./model.py) - One of the core modules to design the architecture for the gans models, be it generator or discriminator.
- [losses](./losses.py) - This module describes losses and penalties for gans such as wasserstien, dcgan and lsgans
- [core](./train_ops) - This is one of the core module which ensures all the pieces are tied together for the models. This is the module which holds all the basic training ops and their extensions based on losses.

## How to train a model?
Training a GANs model consisit of following steps
1. Specify the inputs for your models and morph them into `tf.data.Dataset` instance for feeding it to trainer.
1. Defining a generator model and discriminative model. Currently, we provide abstraction for convolutional models only i.e., `ConvolutionalGeneratorModel` and `ConvolutionalDiscriminatorModel`. You can use any other model too which is derived from `tf.keras.Model`.
1. Select your optimizers for training ops from `tf.optimizers`
1. Based on your loss function you can select any of the trainer. Currently we providing training using Wasserstien, DCGAN and LSGAN based learning strategies. If you want to create your own custom trainer then you can derive a class from `BaseGANTrainer` and override `train`, `train_step` and `__init__` methods.

At this stage you can either use GANs classes for abstraction for your convenience or provision your own for fine-grained control.

## Examples
This section provide some basic example to use this library for training your gans based ops.

* Wasserstien GAN without any checkpoints or saving images after every iterations with default `filters`, `strides` and `kernel_size`
```python
import tensorflow as tf # for tf related ops

from gans.models import ConvolutionalGeneratorModel, ConvolutionalDiscriminativeModel # importing models
from train_ops import WasserstienGANTrainer # for perfroming training ops

# load your dataset here, here for sake of simplicity we are naming it as dataset

# instiantiating generator and discriminator models for the gans
generator = ConvolutionalGeneratorModel(filters=[256, 128, 128, 64],
                                        shape=(7, 7))
discriminator = ConvolutionalDiscriminativeModel(filters=[64, 128])

# defining optimizers
generator_optimizer = tf.optimizers.Adam(learning_rate=1e-4)
discriminator_optimizer = tf.optimizers.Adam(learning_rate=1e-4)

# defining trainer instance
trainer = WasserstienGANTrainer(
    generator_model=generator,
    discriminator_model=discriminator,
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    save_images=False)

# perform training ops
trainer.train(dataset, batch_size=256, noise_dim=100, epochs=100)
```

* DCGAN based learning for 3 dim images with saving checkpoints at every 10th epoch and saving generated images using a seed for vizualizing gans performance. Here we are going to use `kernel_size` of 3x3 and update the shape from 7x7 to 8x8 to generate images of 128x128 dims. 

```python
import tensorflow as tf  # for tf related ops

from gans.models import (  # import models for ops
    ConvolutionalDiscriminativeModel, ConvolutionalGeneratorModel)
from train_ops import DCGANTrainer  # DCGAN trainer

# perfrom dataset ops here

# instiantiating generator and discriminator models for the gans
generator = ConvolutionalGeneratorModel(filters=[512, 256, 256, 128, 64, 32],
                                        shape=(8, 8),
                                        channel_dim=3,
                                        strides=2,
                                        kernel_size=(3, 3))
discriminator = ConvolutionalDiscriminativeModel(filters=[64, 64, 128],
                                                 kernel_size=(3, 3),
                                                 dropout_rate=0.4)

# defining optimizers
generator_optimizer = tf.optimizers.RMSprop(learning_rate=1e-4)
discriminator_optimizer = tf.optimizers.RMSprop(learning_rate=1e-4)

# defining trainer instance
trainer = DCGANTrainer(generator_model=generator,
                       discriminator_model=discriminator,
                       generator_optimizer=generator_optimizer,
                       discriminator_optimizer=discriminator_optimizer,
                       save_checkpoint_at=10,
                       checkpoint_dir="./training_checkpoints",
                       save_images=True)

# perform training ops
trainer.train(dataset, batch_size=256, noise_dim=100, epochs=100)
```

* Using LSGAN based learning strategy to generate 28x28x1 dim images without any middle checkpoint and showing saved images in the process. Instead we will be saving the model at the end of all the training process.
```python
import tensorflow as tf  # for tf related ops

from gans.models import (  # import models for ops
    ConvolutionalDiscriminativeModel, ConvolutionalGeneratorModel)
from train_ops import LSGANTrainer  # LSTrainer trainer

# perfrom dataset ops here

# instiantiating generator and discriminator models for the gans
generator = ConvolutionalGeneratorModel(filters=[256, 128, 128],
                                        shape=(7, 7),
                                        kernel_size=(5, 5))
discriminator = ConvolutionalDiscriminativeModel(filters=[64, 128],
                                                 kernel_size=(5, 5))

# defining optimizers
generator_optimizer = tf.optimizers.RMSprop(learning_rate=1e-4)
discriminator_optimizer = tf.optimizers.RMSprop(learning_rate=1e-4)

# defining trainer instance
trainer = LSGANTrainer(generator_model=generator,
                       discriminator_model=discriminator,
                       generator_optimizer=generator_optimizer,
                       discriminator_optimizer=discriminator_optimizer,
                       save_images=True)

# perform training ops
trainer.train(dataset, batch_size=256, noise_dim=100, epochs=100, show_image=False)

# save the model after training under dir lsgan
trainer.save_checkpoint("lsgan")
```


#### Data Loading Pipelines
This section shows an example for loading data pipelines

* Using images stored in image directory with `batch_size` of 256, `cache` and `prefetch`.

Data must be stored in following fashion
```shell
images_dir
    |- image1.png
    |- image2.png
```

code for reteriving data from above directory
```python
from datapipeline.data_loader import FileDataLoader

import tensorflow as tf

#AutoTune defined
AUTOTUNE = tf.data.AUTOTUNE

# create an instance of Filedataloader
data_handler = FileDataLoader(path_to_images="image_dir",
                              image_extension="png",
                              image_dims=(64, 64),
                              image_channels=3)

# create a dataset of images in batch of 256, shuffled with prefetched and cache
data_handler.create_dataset(batch_size=256,
                            shuffle=True,
                            autotune=AUTOTUNE,
                            cache=True,
                            prefetch=AUTOTUNE)

```

### Author
Sanidhya Mangal, mangalsanidhya19@gmail.com, [website](https://sanidhyamangal.github.io)