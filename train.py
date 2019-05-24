from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import multi_gpu_model
from baseline import baseline
from dataset import Dataset
from generator import Generator
import tensorflow as tf
import time

BATCH_SIZE = 128
VAL_SIZE = 100

# dataset

# dataset_train,num_train = MVBDataset(mode='train', preview=True, shuffle=True, batch_size=BATCH_SIZE)
train_gen = Generator(mode='train')
# print 2 outputs from our generator just to see that it works:
# iter = generator.get_next()
# for i in range(2):
#     print(next(iter))
train = Dataset(train_gen,batch_size=BATCH_SIZE)
dataset_train = train.data
num_train = train.img_count
# model_input = ds.next_element
# print(model_input)
# model = Model(model_input)

test_gen = Generator(mode='test')
test = Dataset(test_gen,batch_size=BATCH_SIZE)
dataset_test = test.data
num_test = test.img_count



# dataset_val = dataset_train.take(VAL_SIZE)
# dataset_train = dataset_train.skip(VAL_SIZE)
# num_train -= VAL_SIZE
# dataset_test,num_test = MVBDataset(mode='test', preview=True, shuffle=True, batch_size=BATCH_SIZE)
# print('SIZES:',num_train,VAL_SIZE,num_test)
# min dataset
# num_train = 3000
# dataset_train = dataset_train.take(num_train)
# dataset_val = dataset_train.skip(num_train).take(VAL_SIZE)
# num_test = 300
# dataset_test = dataset_train.skip(num_train).skip(VAL_SIZE).take(num_test)

steps_per_epoch = num_train//BATCH_SIZE

# model

model = baseline()
# model = multi_gpu_model(model, gpus=2)
model.compile(loss=losses.binary_crossentropy, 
                optimizer=optimizers.Adam(lr=0.001), 
                metrics=['accuracy'])
print(model.summary())
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)  

# filepath = '../checkpoints/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5'

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  # tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='/output', histogram_freq=0, 
                                  embeddings_freq=0, update_freq='batch'),
  # tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', 
  #                                       verbose=1, save_best_only=False, 
  #                                       mode='auto', period=1)
]

model.fit(dataset_train, epochs=1, steps_per_epoch=num_train//BATCH_SIZE, callbacks=callbacks)
            # validation_data=dataset_val, validation_steps=VAL_SIZE//BATCH_SIZE)
 
# EVALUATION
print('### EVALUATION ###')
model.evaluate(dataset_test,steps=num_test//BATCH_SIZE)