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