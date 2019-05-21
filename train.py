from tensorflow.keras import losses, optimizers
from baseline import baseline
from dataset import MVBDataset
import tensorflow as tf

BATCH_SIZE = 128

data = MVBDataset(preview=True, shuffle=True, batch_size=BATCH_SIZE)

model = baseline()
model.compile(loss=losses.binary_crossentropy, 
                optimizer=optimizers.Adam(lr=0.001), 
                metrics=['accuracy'])

