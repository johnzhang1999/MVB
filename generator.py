import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pathlib

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Generator(object):
  def __init__(self,mode='train',path='../data/MVB_0505'):
    self.lib = self.generate_lib(path,mode)

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

      yield (p,g),vec