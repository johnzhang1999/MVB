import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pathlib
from generator import Generator

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

class Dataset(object):

  def __init__(self, generator=Generator(mode='train'), preview=False, 
                  prefetch=True, batch_size=32):
    self.preview = preview
    self.prefetch = prefetch
    self.batch_size = batch_size
    self.data,self.img_count = self.MVBDataset(generator)

  def _preprocess_image(self, image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [256, 256])
    image /= 255.0  # normalize to [0,1] range
    return image

  def _load_and_preprocess_image(self,paths,label):
    a = tf.read_file(paths[0])
    b = tf.read_file(paths[1])
    return (self._preprocess_image(a),self._preprocess_image(b)),label

  def MVBDataset(self, generator: Generator):
    lib = generator.lib
    img_count = len(lib.items())

    # building a tf dataset
    dataset = tf.data.Dataset.from_generator(generator.get_next, 
                output_types=((tf.string, tf.string),tf.float16))

    dataset = dataset.map(self._load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    print('shape: ', repr(dataset.output_shapes))
    print('type: ', dataset.output_types)
    print()
    print(dataset)

    # visual examinations
    if self.preview: self.preview_dataset(dataset)

    # repeat, batch and prefetch
    ds = dataset.repeat()
    ds = ds.batch(self.batch_size)
    if self.prefetch:
      ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds,img_count

  def preview_dataset(self, dataset, num=4, save=False, path='previews/ds.png'):
    plt.figure(figsize=(8,2*num))
    for n,((a,b),m) in enumerate(dataset.take(num)):
      plt.subplot(4,2,2*n+1)
      plt.imshow(a)
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      plt.subplot(4,2,2*n+2)
      plt.imshow(b)
      plt.grid(False)
      plt.xticks([])
      plt.yticks([])
      plt.xlabel(str(m.numpy()))
    plt.show()
    if save: plt.savefig(path)