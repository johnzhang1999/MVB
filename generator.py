import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Generator(object):
  def __init__(self,mode='train',path='../data/MVB_0505'):
    self.lib = self.generate_lib(path,mode)
    self.aug_gen = ImageDataGenerator(rotation_range=15,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=True,
                               vertical_flip=False,
                               fill_mode='nearest',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5])

  def generate_lib(self,path, mode):
    lib = {}
    data_root = pathlib.Path(path) # CHANGE min_data to data for full dataset
    if mode == 'test':
      mode = 'test_500'
    train_path = data_root / mode
    for bag in train_path.iterdir():
      id = str(bag.stem)
      if '.DS_Store' in id: continue
      lib[id] = {}
      for dir in bag.iterdir():
        result = list(dir.glob('*.jpg'))
        result = [str(img) for img in result]
        if str(dir.stem)[0] == 'b':
          lib[id]['gallery'] = result
        if str(dir.stem)[0] == 'f':
          lib[id]['probe'] = result
    # print(lib)
    return lib

  def _preprocess_image(self, image):
    image = plt.imread(image)
    image = self.aug_gen.random_transform(image)
    image = tf.image.resize_images(image, [256, 256])
    image /= 255.0 # normalize to [0,1] range
    return image

  def get_next(self):
    lib = list(self.lib.keys())
    while True:
      probe = random.choice(lib)
      same_bag = random.random() > 0.5
      if same_bag:
        gallery = probe
        vec = [1.0,0.0]
      else:
        gallery = random.choice(lib)
        vec = [0.0,1.0]
      
      p = random.choice(self.lib[probe]['probe'])
      g = random.choice(self.lib[gallery]['gallery'])

      p,g = self._preprocess_image(p),self._preprocess_image(g)

      yield (p,g),vec