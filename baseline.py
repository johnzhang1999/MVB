from tensorflow.keras import models, layers, activations, losses, optimizers
import tensorflow.keras.backend as K
import tensorflow as tf

# Model configs
DIMEN = 128 

input_shape = ((DIMEN**2) * 3,)
convolution_shape = (DIMEN,DIMEN,3)
kernel_size_1 = (4,4)
kernel_size_2 = (3,3)
pool_size_1 = (3,3)
pool_size_2 = (2,2)
strides = 2

p_input = layers.Input(shape=input_shape)
g_input = layers.Input(shape=input_shape)


# Conv layers begin
# Conv 1-3

shared_blue = [
	# Reshape(input_shape=input_shape , target_shape=convolution_shape),
	layers.Conv2D(32, kernel_size=kernel_size_1, strides=strides, activation='relu'),
    layers.MaxPooling2D(pool_size=pool_size_1, strides=strides),
]
BlueUnit1 = tf.keras.Sequential(shared_blue)
BlueUnit2 = tf.keras.Sequential(shared_blue)
BlueUnit3 = tf.keras.Sequential(shared_blue)

p = BlueUnit1(p_input)
g = BlueUnit1(g_input)
p = BlueUnit2(p)
g = BlueUnit2(g)
p = BlueUnit3(p)
g = BlueUnit3(g)

# Conv4

conv1 = layers.Conv2D(32, kernal_size=kernal_size_1, strides=strides, activation='relu')
p_bn1 = layers.batch_normalization()
g_bn1 = layers.batch_normalization()

p = conv1(p)
g = conv1(g)
p = p_bn1(p)
g = g_bn1(g)

shared_green = [
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=pool_size_1, strides=strides),
]

GreenUnit1 = tf.keras.Sequential(shared_green)

p = GreenUnit1(p)
g = GreenUnit1(g)

# SEBlock Begins here
p_channel = K.int_shape(p)[-1]
g_channel = K.int_shape(g)[-1]
# assert p_channel == g_channel
ratio = 16

shared_se = [
    layers.GlobalAveragePooling2D(),
    layers.Dense(p_channel // ratio, activation='relu'),
    layers.Dense(p_channel, activation='sigmoid'),
]

SE1 = tf.keras.Sequential(shared_se)
p_se = SE1(p)
g_se = SE2(g)
p = layers.Multiply()([p, p_se])
g = layers.Multiply()([g, g_se])

# Conv5

conv2 = layers.Conv2D(32, kernal_size=kernal_size_1, strides=strides, activation='relu')
p_bn2 = layers.batch_normalization()
g_bn2 = layers.batch_normalization()

p = conv2(p)
g = conv2(g)
p = p_bn2(p)
g = g_bn2(g)

GreenUnit2 = tf.keras.Sequential(shared_green)
p = GreenUnit2(p)
g = GreenUnit2(g)

SE2 = tf.keras.Sequential(shared_se)
p_se = SE2(p)
g_se = SE2(g)
p = layers.Multiply()([p, p_se])
g = layers.Multiply()([g, g_se])

# Conv layers finished

# Element-wise subtraction
sub = layers.Subtraction()([p,g])

# Dense layers
x = layers.Dense(64, activation='sigmoid')(sub)
x = layers.Dense(16, activation='sigmoid')(sub)
output = layers.Dense(1, activation='sigmoid')(x)

# Model Compilation
model = models.Model([p_input,g_input],output)
model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])

