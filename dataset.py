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
    # # generating positive pairs
    # pos = []

    # for img,val_dict in lib.items():
    #     ps = val_dict['probe']
    #     gs = val_dict['gallery']
    #     for p in ps:
    #         for g in gs:
    #             pos.append((p,g)) # CHECK: tuple or list?
    # print('pos pair num:', len(pos))

    # # generating negative pairs

    # # ENSURE a balanced pos and neg pair ratio for training
    # neg_sample_num = len(pos) // len(lib.keys())
    # neg = []

    # def get_random_not(lib,img):
    #     k,v = random.choice(list(lib.items()))
    #     if k == img: get_random_not(lib,img)
    #     return v

    # for img,val_dict in lib.items():
    #     ps =  val_dict['probe']
    #     for i in range(neg_sample_num):
    #         rand_neg_gs = get_random_not(lib,img)['gallery']
    #         neg.append((random.choice(ps),random.choice(gs)))

    # print('neg pair num:', len(neg))

    # building a tf dataset
    dataset = tf.data.Dataset.from_generator(generator.get_next, 
                output_types=((tf.string, tf.string),tf.int8))


    # # building tf dataset
    # pos_labels = np.ones(len(pos))
    # neg_labels = np.zeros(len(neg))
    # labels = np.concatenate([pos_labels,neg_labels])
    # label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    # # label_ds = tf.data.Dataset.from_tensor_slices(labels)
    # for label in label_ds.take(10):
    #   print(label.numpy())

    # imgs = pos + neg
    # # print(imgs[0][0])
    # img_count = len(imgs)
    # img_ds = tf.data.Dataset.from_tensor_slices(imgs)
    dataset = dataset.map(self._load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    print('shape: ', repr(dataset.output_shapes))
    print('type: ', dataset.output_types)
    print()
    print(dataset)
    
    # # zipping the labels and images
    # image_label_ds = tf.data.Dataset.zip((img_ds, label_ds))
    # print('image shape: ', image_label_ds.output_shapes[0])
    # print('label shape: ', image_label_ds.output_shapes[1])
    # print('types: ', image_label_ds.output_types)
    # print()
    # print(image_label_ds)

    # visual examinations
    if self.preview: self.preview_dataset(dataset)
      
    
    # ds = image_label_ds.cache()
    # ds = dataset
    # ds = ds.apply(
    #   tf.data.experimental.shuffle_and_repeat(buffer_size=round(img_count//1.7)))
    # if shuffle:
    #   # can reduce buffer_size if memory runs out
    #   ds = image_label_ds.shuffle(buffer_size=round(img_count/1.5))
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
    plt.savefig(path)