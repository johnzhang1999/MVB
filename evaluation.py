import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import losses, optimizers
from baseline import baseline
from dataset import Dataset
from generator import Generator

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 128

test_gen = Generator(mode='test')
lib = test_gen.lib

filename_lookup = {}
for id,ps_and_gs in lib.items():
    for t,imgs in ps_and_gs.items():
      for img in imgs:
        filename_lookup[img] = id

test = []

for id,ps_and_gs in lib.items():
  for p in ps_and_gs['probe']:
    pairs = []
    for inner_id,inner_ps_and_gs in lib.items():
      for g in inner_ps_and_gs['gallery']:
        pairs.append((p,g))
    answers = ps_and_gs['gallery']
    test.append((id,p,pairs,answers))


model = baseline()
weights_path = '../checkpoints/saved-model-04-0.90.hdf5'
model.load_weights(weights_path)
model.compile(loss=losses.binary_crossentropy, 
        optimizer=optimizers.Adam(lr=0.0001), 
        metrics=['accuracy'])

# procedure: for every img, test all pairs, find the k img(s) with highest scores, see if it's in answers

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [256, 256])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_process_img(paths):
  a = tf.read_file(paths['img_a'])
  b = tf.read_file(paths['img_b'])
  return {'img_a':preprocess_image(a),'img_b':preprocess_image(b)}


# params
total = 20 # total num of test cases
n = 10 # rank-n
ave_top_n = 2 # average top ave_top_n of gallery scores

correct = 0.0
for i in range(total):
  id_truth,p,pairs,_ = random.choice(test)
  pairs_list = pairs
  print('###Number of pairs:',len(pairs_list))
  first = [pair[0] for pair in pairs]
  second = [pair[1] for pair in pairs]
  pairs = tf.data.Dataset.from_tensor_slices({'img_a':first,'img_b':second})

  pairs = pairs.map(load_and_process_img, num_parallel_calls=AUTOTUNE)
  pairs = pairs.batch(BATCH_SIZE).prefetch(AUTOTUNE)

  # iterator = pairs.make_one_shot_iterator()
  # print('iterator finished')
  # print(iterator.get_next())


  # THOUGHTS: get a file-score pair list, then sum all scores of a single image

  predictions = model.predict(pairs,steps=len(pairs_list)//BATCH_SIZE+1,verbose=1,)
  print('###LEN###:',len(predictions))
  similarity_scores = predictions[:,0]
  image_score_dict = dict(zip(second,similarity_scores))
  bag_score_dict = {}
  for id,ps_and_gs in lib.items():
    all_scores = np.array(list(map(lambda x:image_score_dict[x], ps_and_gs['gallery'])))
    if len(all_scores) > ave_top_n:
      top_two_scores_idx = np.argpartition(all_scores, -ave_top_n)[-ave_top_n:]
      top_two_scores = all_scores[top_two_scores_idx]
    else: 
      top_two_scores = all_scores
    bag_score_dict[id] = np.average(top_two_scores)
    # print(bag_score_dict[id])
  bag_scores = np.array(list(bag_score_dict.values()))
  bags = list(bag_score_dict.keys())
  top_n_idx = np.argpartition(bag_scores, -n)[-n:]    
  top_n_idx = top_n_idx[np.argsort((-bag_scores)[top_n_idx])] # sorted
  top_n_matches = np.array(bags)[top_n_idx]
  top_n_matches_and_scores = dict(zip(top_n_matches,bag_scores[top_n_idx]))
  print('top {} matches are: {}'.format(n,top_n_matches_and_scores))
  # top_n_matches = [p[1] for p in top_n_matches]
  # print(predictions[highest,0])
  mark = id_truth in top_n_matches
  if mark:
    correct += 1
    idx = np.where(top_n_matches==id_truth)[0][0]+1
    print('Yes! Truth is ranked:', idx)
  else: 
    print('No :(')
  print('answer:',id_truth)
  print('current score ({}/{} test): {}'.format(i+1,total,correct/(i+1)))

print('###final accuracy###:',correct/total)
