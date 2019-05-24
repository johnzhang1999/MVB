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
model.load_weights('../checkpoints/saved-model-70-0.87.hdf5')
model.compile(loss=losses.binary_crossentropy, 
                optimizer=optimizers.Adam(lr=0.0001), 
                metrics=['accuracy'])

def first(self,t):
    return t[0]

# lib = test_gen.lib
# p = list(lib.values())[0]['probe'][0]
# g = list(lib.values())[0]['gallery'][0]
# print(p,g)
# p,_ = test._load_and_preprocess_image(p,0)
# g,_ = test._load_and_preprocess_image(g,0)
# tensor = [p,g]

dataset = dataset_test.map(first, num_parallel_calls=AUTOTUNE)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(1):
  print(next_element.numpy())




1/0
result = model.predict(test_set,batch_size=BATCH_SIZE,steps=1)
print ('result###',result.numpy())
