import tensorflow as tf
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    # answers = ps_and_gs['gallery']
    test.append((id,p,pairs))


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
total = 10 # total num of test cases
n = 3 # rank-n
ave_top_n = 3 # average top ave_top_n of gallery scores

correct = 0.0
correct_preds = [] # correct
incorrect_preds = {} # incorrect
for i in range(total):
  id_truth,p,pairs = random.choice(test)
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
    assert list(top_n_matches_and_scores.items())[idx-1][0] == id_truth
    correct_preds.append(list(top_n_matches_and_scores.items())[idx-1])
  else: 
    print('No :(')
    incorrect_preds[id_truth] = list(top_n_matches_and_scores.items())[0]
  print('answer:',id_truth)
  print('current score ({}/{} test): {}'.format(i+1,total,correct/(i+1)))
acc = correct/total
print('###final accuracy###:',acc)

# logging
with open('evals.log', 'a') as f:
  localtime = time.asctime(time.localtime(time.time()))
  f.write('Run at: {}\n'.format(localtime))
  f.write('Using model: {}\n'.format(weights_path))
  f.write('Tests #: {}\n'.format(total))
  f.write('Using top: {}\n'.format(ave_top_n))
  f.write('Hitting rank: {}\n'.format(n))
  f.write('Accuracy: {}\n\n'.format(acc))

# plotting correct predicts
correct = int(correct)
plt.figure(figsize=(5,5*total))
plt.title('Predictions')
for n,(img_id,score) in enumerate(correct_preds):
  plt.subplot(total,1,n+1)
  img = mpimg.imread(random.choice(lib[img_id]['probe']))
  plt.imshow(img)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel('{}\n{}'.format(img_id,score))

# plotting incorrect predicts
incorrect = len(list(incorrect_preds.keys()))
for n,(truth,pred) in enumerate(list(incorrect_preds.items())):
  plt.subplot(total,2,2*correct+2*n+1)
  img = mpimg.imread(random.choice(lib[truth]['probe']))
  plt.imshow(img)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(truth)
  plt.subplot(total,2,2*correct+2*n+2)
  img = mpimg.imread(random.choice(lib[pred[0]]['probe']))
  plt.imshow(img)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel('{}\n{}'.format(pred[0],pred[1]))
path = 'predictions-t{}-r{}-a{}.png'.format(str(time.time()),str(n),str(acc))
plt.savefig(path)
print('plot saved.')