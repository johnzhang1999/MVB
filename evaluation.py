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

lib = test_gen.lib
p = list(lib.values())[0]['probe'][0]
g = list(lib.values())[0]['gallery'][0]
print(p,g)
p,_ = test._load_and_preprocess_image(p,0)
g,_ = test._load_and_preprocess_image(g,0)
tensor = [p,g]

# def _preprocess_image(image):
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize_images(image, [256, 256])
#     image /= 255.0  # normalize to [0,1] range
#     return image

# def _load_and_preprocess_image(t):
#     paths=t[0]
#     label=t[1]
#     a = tf.read_file(paths[0])
#     b = tf.read_file(paths[1])
#     return ([_preprocess_image(a)],[_preprocess_image(b)])
# iter = test_gen.get_next()
# tensor = next(iter)
# print(type(tensor))
# tensor = _load_and_preprocess_image(tensor)
# # print(tensor)

result = model.predict(tensor,batch_size=BATCH_SIZE,steps=1)
print ('result###',result.numpy())
