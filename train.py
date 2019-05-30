from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import multi_gpu_model
from baseline import baseline
from ft_resnet import ft_resnet
from dataset import Dataset
from generator import Generator
import tensorflow as tf
import time
import os

tf.enable_eager_execution()


BATCH_SIZE = 64
SPLIT_RATIO = 0.2

# dataset

train_gen = Generator(mode='train')
# print 2 outputs from our generator just to see that it works:
# iter = generator.get_next()
# for i in range(2):
#     print(next(iter))
dataset = Dataset(train_gen,augment=True,train_val_split=SPLIT_RATIO,
          batch_size=BATCH_SIZE,preview=True)
dataset_val = dataset.val_dataset
dataset_train = dataset.train_dataset
VAL_SIZE = dataset.val_count
num_train = dataset.train_count

test_gen = Generator(mode='test')
test = Dataset(test_gen,batch_size=BATCH_SIZE)
dataset_test = test.train_dataset # sorry for the confusing naming here...
num_test = test.train_count

# model

model = ft_resnet()
# model = multi_gpu_model(model, gpus=2)
# model.load_weights('../checkpoints/saved-model-151-0.91.hdf5')
model.compile(loss=losses.binary_crossentropy, 
        optimizer=optimizers.Adam(lr=0.0001), 
        metrics=['accuracy'])
print(model.summary())
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)  

filepath = '../checkpoints/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5'
if not os.path.exists('../checkpoints'):
  os.makedirs('../checkpoints')

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='/output/resnet5-none-weights', histogram_freq=0, 
                  embeddings_freq=0, update_freq='batch'),
  tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', 
                    verbose=1, save_best_only=True, 
                    save_weights_only=True,
                    mode='auto', period=1)
]

model.fit(dataset_train, epochs=200, steps_per_epoch=num_train//BATCH_SIZE, callbacks=callbacks,
      validation_data=dataset_val, validation_steps=VAL_SIZE//BATCH_SIZE)
 
# EVALUATION
print('### EVALUATION ###')
model.evaluate(dataset_test,steps=num_test//BATCH_SIZE)
