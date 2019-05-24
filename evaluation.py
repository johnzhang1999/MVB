import tensorflow as tf
from tensorflow.keras import losses, optimizers
from baseline import baseline
from dataset import Dataset
from generator import Generator

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 1

test_gen = Generator(mode='test')
test = Dataset(test_gen,batch_size=BATCH_SIZE)
dataset_test = test.data
num_test = test.img_count

model = baseline()
model.load_weights('../checkpoints/saved-model-01-0.48.hdf5')
model.compile(loss=losses.binary_crossentropy, 
                optimizer=optimizers.Adam(lr=0.0001), 
                metrics=['accuracy'])

def first(pair,l):
    return pair

# lib = test_gen.lib
# p = list(lib.values())[0]['probe'][0]
# g = list(lib.values())[0]['gallery'][0]
# print(p,g)
# p,_ = test._load_and_preprocess_image(p,0)
# g,_ = test._load_and_preprocess_image(g,0)
# tensor = [p,g]
dataset = dataset_test
print('###############')
print('shape: ', repr(dataset.output_shapes))
print('type: ', dataset.output_types)
print()
print(dataset)
print('###############')
dataset = dataset_test.map(first, num_parallel_calls=AUTOTUNE)
print('shape: ', repr(dataset.output_shapes))
print('type: ', dataset.output_types)
print()
print(dataset)
print('###############')
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
print(next_element)
result = model.predict(next_element,steps=1,verbose=1)
print ('result1###',result)
result = model.predict(next_element,steps=1,verbose=1)
print ('result2###',result)