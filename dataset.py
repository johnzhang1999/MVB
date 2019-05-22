import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pathlib

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [256, 256])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(paths):
  a = tf.read_file(paths[0])
  b = tf.read_file(paths[1])
  return preprocess_image(a),preprocess_image(b)

def MVBDataset(path='../data/MVB_0505', mode='train', preview=False, 
                shuffle=True, prefetch=True, batch_size=32):
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

  # generating positive pairs
  pos = []

  for img,val_dict in lib.items():
      ps = val_dict['probe']
      gs = val_dict['gallery']
      for p in ps:
          for g in gs:
              pos.append((p,g)) # CHECK: tuple or list?
  print('pos pair num:', len(pos))

  # generating negative pairs

  # ENSURE a balanced pos and neg pair ratio for training
  neg_sample_num = len(pos) // len(lib.keys())
  neg = []

  def get_random_not(lib,img):
      k,v = random.choice(list(lib.items()))
      if k == img: get_random_not(lib,img)
      return v

  for img,val_dict in lib.items():
      ps =  val_dict['probe']
      for i in range(neg_sample_num):
          rand_neg_gs = get_random_not(lib,img)['gallery']
          neg.append((random.choice(ps),random.choice(gs)))

  print('neg pair num:', len(neg))

  
  # building tf dataset
  pos_labels = np.ones(len(pos))
  neg_labels = np.zeros(len(neg))
  labels = np.concatenate([pos_labels,neg_labels])
  label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
  # label_ds = tf.data.Dataset.from_tensor_slices(labels)
  for label in label_ds.take(10):
    print(label.numpy())

  imgs = pos + neg
  # print(imgs[0][0])
  img_count = len(imgs)
  img_ds = tf.data.Dataset.from_tensor_slices(imgs)
  img_ds = img_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
  print('shape: ', repr(img_ds.output_shapes))
  print('type: ', img_ds.output_types)
  print()
  print(img_ds)
  
  # zipping the labels and images
  image_label_ds = tf.data.Dataset.zip((img_ds, label_ds))
  print('image shape: ', image_label_ds.output_shapes[0])
  print('label shape: ', image_label_ds.output_shapes[1])
  print('types: ', image_label_ds.output_types)
  print()
  print(image_label_ds)

  # visual examinations
  if preview: 
    plt.figure(figsize=(8,8))
    for n,((a,b),m) in enumerate(image_label_ds.shuffle(200).take(4)):
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
  
  ds = image_label_ds.cache(filename='./cache.tf-data')
  ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=round(img_count/1.5)))
  # if shuffle:
  #   # can reduce buffer_size if memory runs out
  #   ds = image_label_ds.shuffle(buffer_size=round(img_count/1.5))
  # ds = ds.repeat()
  ds = ds.batch(batch_size)
  if prefetch:
    ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds,img_count