import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pathlib

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [256, 256])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

lib = {}
data_root = pathlib.Path('../min_data/MVB_0505') # CHANGE min_data to data for full dataset
train_path = data_root / 'train'
for bag in train_path.iterdir():
    id = str(bag.stem)
    if '.DS_Store' in id: continue
    lib[id] = {}
    for dir in bag.iterdir():
        result = list(dir.glob('*.jpg'))
        result = [load_and_preprocess_image(str(img)) for img in result]
        if str(dir.stem)[0] == 'b':
            lib[id]['gallery'] = result
        if str(dir.stem)[0] == 'f':
            lib[id]['probe'] = result
# print(lib)

pos = []
# generating positive pairs
for img,val_dict in lib.items():
    ps = val_dict['probe']
    gs = val_dict['gallery']
    for p in ps:
        for g in gs:
            pos.append((ps,gs)) # CHECK: tuple or list?
print('pos pair num:', len(pos))
# ENSURE a balanced pos and neg pair ratio for training
neg_sample_num = len(pos) // len(lib.keys())
neg = []

def get_random_not(lib,img):
    k,v = random.choice(list(lib.items()))
    if k == img: get_random_not(lib,img)
    return v

# generating negative pairs
for img,val_dict in lib.items():
    ps =  val_dict['probe']
    for i in range(neg_sample_num):
        rand_neg_gs = get_random_not(lib,img)['gallery']
        neg.append((random.choice(ps),random.choice(gs)))

print('neg pair num:', len(neg))
# print(neg)