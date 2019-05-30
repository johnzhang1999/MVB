import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pathlib
import os
from generator import Generator

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset(object):

  def __init__(self, generator=Generator(mode='train'), augment=False, preview=False, 
          prefetch=True, batch_size=32, train_val_split=0):
    self.augment = augment
    self.preview = preview
    self.prefetch = prefetch
    self.batch_size = batch_size
    self.train_val_split = train_val_split
    self.total_count = 0
    self.train_count = 0
    self.val_count = 0
    self.train_dataset, self.val_dataset = self.MVBDataset(generator)
    if self.train_val_split == 0:
      assert self.val_count == None and self.val_dataset == None
    else:
      assert self.val_count != None and self.val_dataset != None

  def _preprocess_image(self, image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [256, 256])
    image /= 255.0 # normalize to [0,1] range
    return image

  def _load_and_preprocess_image(self,paths,label):
    a = tf.read_file(paths[0])
    b = tf.read_file(paths[1])
    return (self._preprocess_image(a),self._preprocess_image(b)),label

  def MVBDataset(self, generator: Generator):
    lib = generator.lib
    self.total_count = len(lib.items())

    # building a tf dataset
    dataset = tf.data.Dataset.from_generator(generator.get_next, 
          output_types=((tf.string, tf.string),tf.float16))

    # dataset processing
    dataset = dataset.map(self._load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    
    # splitting train and val
    if self.train_val_split != 0:
      self.val_count = round(self.total_count * self.train_val_split)
      self.train_count = self.total_count - self.val_count
      val_dataset = dataset.take(self.val_count)
      dataset = dataset.skip(self.val_count)
    else:
      self.val_count = None
      self.train_count = self.total_count
      val_dataset = None

    # Add augmentations if needed
    if self.augment:
      augmentations = [self.flip, self.color, self.zoom, self.rotate]
      for f in augmentations:
        dataset = dataset.map(lambda x,y: ((tf.cond(tf.random_uniform([], 0, 1) > 0, 
                    lambda: f(x[0]), lambda: x[0]), tf.cond(tf.random_uniform([], 0, 1) > 0, 
                    lambda: f(x[1]), lambda: x[1])),y), num_parallel_calls=AUTOTUNE)
      dataset = dataset.map(lambda x,y: ((tf.clip_by_value(x[0], 0, 1),tf.clip_by_value(x[1], 0, 1)),y))

    # sanity check
    print('shape: ', repr(dataset.output_shapes))
    print('type: ', dataset.output_types)
    print()
    print(dataset)

    # visual examinations
    if self.preview: self.preview_dataset(dataset,save=True)

    # repeat, batch and prefetch
    ds = dataset.repeat()
    ds = ds.batch(self.batch_size)
    if self.prefetch:
      ds = ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = val_dataset
    if self.train_val_split != 0:
      val_ds = val_dataset.repeat()
      val_ds = val_ds.batch(self.batch_size)
      if self.prefetch:
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return ds,val_ds

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
    if save: 
      if not os.path.exists('previews'):
        os.makedirs('previews')
      plt.savefig(path)

  # Source: https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
  def flip(self, x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
      x: Image to flip

    Returns:
      Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

  def color(self, x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
      x: Image

    Returns:
      Augmented image
    """
    # x = tf.image.random_hue(x, 0.08)
    # x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

  def rotate(self, x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
      x: Image

    Returns:
      Augmented image
    """

    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

  def zoom(self, x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
      x: Image

    Returns:
      Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
      x1 = y1 = 0.5 - (0.5 * scale)
      x2 = y2 = 0.5 + (0.5 * scale)
      boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
      # Create different crops for an image
      crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(256, 256))
      # Return a random crop
      return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))