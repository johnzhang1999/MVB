from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model
from baseline import baseline
from dataset import MVBDataset
import tensorflow as tf

BATCH_SIZE = 128

dataset_train = MVBDataset(mode='train', preview=False, shuffle=True, batch_size=BATCH_SIZE)
dataset_train = dataset_train.take(1000)
dataset_val = dataset_train.skip(1000)
dataset_test = MVBDataset(mode='test', preview=False, shuffle=True, batch_size=BATCH_SIZE)

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
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(dataset, epochs=10, steps_per_epoch=30, 
            callbacks=callbacks, validation_data=dataset_val)