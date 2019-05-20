import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import random
import pathlib

lib = {}
data_root = pathlib.Path('../data/MVB_0505')
train_path = data_root / 'train'
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
print(lib)

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [128, 128])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)
