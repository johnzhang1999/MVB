from tensorflow.keras import losses, optimizers
from tensorflow.keras.utils import plot_model
from baseline import baseline
from dataset import MVBDataset
import tensorflow as tf

BATCH_SIZE = 128

# data = MVBDataset(preview=True, shuffle=True, batch_size=BATCH_SIZE)

model = baseline()
model.compile(loss=losses.binary_crossentropy, 
                optimizer=optimizers.Adam(lr=0.001), 
                metrics=['accuracy'])
print(model.summary())
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)  

