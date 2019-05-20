from tensorflow.keras import losses, optimizers
from baseline import baseline
import tensorflow as tf

model = baseline()
model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])