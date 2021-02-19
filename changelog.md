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
- fixed checkpoint issues, earlier was recording losses instead of optimizers hence fixed it.

### 2020-12-22
- feat added dcgan trainer for training DCGAN

### 2020-12-23
- feat added lsgan based trainer and mixin for the ops
- generalized criterion based loss for the lsgan and dcgan

### 2020-12-26
- added save_checkpoint method in the base trainer method to save checkpoint in any state
- migrated epochs from constructor to train method
- added param for showing image in train method
- renamed namedtuples to train_ops

### 2020-12-27
- added readme for better description

### 2020-12-31
- added datapipeline and data loader from files
- updated readme for data loading pipelines

### 2021-01-02
- fixed: issues with file data loader

### 2021-01-07
- fixed: checkpoint issue for saving
- fixed: multiple issues from deepsource such as docs, pass, etc

### 2021-01-09
- added: resnet module and resent based discriminative models
- packaged the entire application to be deployed on PyPI
- updated readme for including residual framework usage example

### 2021-02-19
- feat: added drop reminder option in data loader
