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
# test = Dataset(test_gen,batch_size=BATCH_SIZE)
lib = test_gen.lib
# dataset_test = test.data
# num_test = test.img_count

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
model.load_weights('../checkpoints/saved-model-41-0.81.hdf5')
model.compile(loss=losses.binary_crossentropy, 
                optimizer=optimizers.Adam(lr=0.0001), 
                metrics=['accuracy'])

# def first(pair,l):
#     return pair

# playing around

# dataset = dataset_test
# print('###############')
# print('shape: ', repr(dataset.output_shapes))
# print('type: ', dataset.output_types)
# print()
# print(dataset)
# print('###############')
# dataset = dataset_test.map(first, num_parallel_calls=AUTOTUNE)
# print('shape: ', repr(dataset.output_shapes))
# print('type: ', dataset.output_types)
# print()
# print(dataset)
# print('###############')
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
# print(next_element)
# result = model.predict(next_element,steps=1,verbose=1)
# print ('result1###',result)
# next_element = iterator.get_next()
# result = model.predict(next_element,steps=1,verbose=1)
# print ('result2###',result)

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


correct = 0.0
total = 20
n = 100

class MiniGenerator(object):
    def __init__(self,list):
        self.list = list
    def get_next(self):
        for elem in self.list:
            yield elem


for i in range(total):
    id,p,pairs,ans = random.choice(test)
    pairs_list = pairs
    first = [pair[0] for pair in pairs]
    second = [pair[1] for pair in pairs]
    pairs = tf.data.Dataset.from_tensor_slices({'img_a':first,'img_b':second})

    # gen = MiniGenerator(pairs)
    # pairs = tf.data.Dataset.from_generator(gen.get_next,output_types=tf.string)

    pairs = pairs.map(load_and_process_img, num_parallel_calls=AUTOTUNE)
    pairs = pairs.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # iterator = pairs.make_one_shot_iterator()
    # print('iterator finished')
    # print(iterator.get_next())
    # 1/0
    predictions = model.predict(pairs,steps=len(pairs_list)//BATCH_SIZE,verbose=1)
    print('###LEN###:',len(predictions))
    similarity_scores = predictions[:,0]
    # highest = np.argmax(similarity_scores)
    top_n_idx = np.argpartition(similarity_scores, -n)[-n:]
    top_n_matches = np.array(pairs_list)[top_n_idx]
    top_n_matches = [p[1] for p in top_n_matches]
    # print(predictions[highest,0])
    mark = not set(top_n_matches).isdisjoint(ans)
    print(mark)
    if mark:
        correct += 1
        print('Yes!')
        # print('Current score:',)

print('###accuracy###:',correct/total)
