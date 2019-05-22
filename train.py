from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model
from baseline import baseline
from dataset import MVBDataset
import tensorflow as tf
import time

BATCH_SIZE = 128
VAL_SIZE = 500

# dataset

dataset_train,num_train = MVBDataset(mode='train', preview=False, shuffle=False, batch_size=BATCH_SIZE)
dataset_val = dataset_train.take(VAL_SIZE)
dataset_train = dataset_train.skip(VAL_SIZE)
num_train -= VAL_SIZE
dataset_test,num_test = MVBDataset(mode='test', preview=False, shuffle=False, batch_size=BATCH_SIZE)
print('SIZES:',num_train,VAL_SIZE,num_test)

steps_per_epoch = num_train//BATCH_SIZE

def timeit(ds, batches=2*steps_per_epoch+1):
  overall_start = time.time()
  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
  # before starting the timer
  it = iter(ds.take(batches+1))
  next(it)
  print('finished fetching the first batch')
  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(batches, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
  print("Total time: {}s".format(end-overall_start))

timeit(dataset_train) # Well, there is a serious problem with dataset performance...
1/0

# model

model = baseline()
model.compile(loss=losses.binary_crossentropy, 
                optimizer=optimizers.Adam(lr=0.001), 
                metrics=['accuracy'])
print(model.summary())
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)  

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='/output', histogram_freq=0, 
                                  batch_size=BATCH_SIZE, write_graph=True, 
                                  write_grads=False, write_images=False, 
                                  embeddings_freq=0, embeddings_layer_names=None, 
                                  embeddings_metadata=None, embeddings_data=None, 
                                  update_freq='batch')
]

model.fit(dataset_train, epochs=5, steps_per_epoch=num_train//BATCH_SIZE, callbacks=callbacks, 
            validation_data=dataset_val, validation_steps=VAL_SIZE//BATCH_SIZE)
 
# EVALUATION
print('### EVALUATION ###')
model.evaluate(dataset_test,steps=num_test//BATCH_SIZE)