### 2020-12-11
- added models file for bootstraping generator and discriminator

### 2020-12-16
- added git origin, changelog and other git files
- baseline train ops for contemplating training using class based ops
- [WIP] need to add save_images_at and save_checkpoint at functions in train ops

### 2020-12-17
- added author information
- added implementation for saving and generation of image ops
- sorted imports
- implemented saving checkpoint for the base trainer models

### 2020-12-18
- fixed wasserstien training ops by changing loss to mixins
- fixed typos in the base training init function
- fixed positional args error in save function
- fixed: mro for wasserstien gans
- upated: seed dims from batch_size to 16 for streamlining the process
- updated: position of saving image to actually save image after epoch

### 2020-12-19
- added: base layer for the first conv with strides values 1
- added init file and relative imports
- removed input layer from the discriminative model
- fixed: train step issues

### 2020-12-21
- fixed issues with trainer module caused by incorrect loss function, made trainer as extension of module class
