from tensorflow.keras import models, layers, activations, losses, optimizers, applications
import tensorflow.keras.backend as K
import tensorflow as tf

def ft_resnet():
  # Model configs
  DIMEN = 256 
  kernel_size_1 = (3,3)
  pool_size_1 = (2,2)
  strides = 1
  input_shape = (DIMEN,DIMEN,3)

  # Input layers
  p_input = layers.Input(shape=input_shape,name='img_a')
  g_input = layers.Input(shape=input_shape,name='img_b')

  # Conv layers begin
  p,g = p_input, g_input
  
  # Load pretrained weights
  resnet_model = applications.ResNet50(include_top=False, weights=None,
                  input_shape=input_shape)
  print(resnet_model.summary())

  # for layer in resnet_model.layers:
  #   layer.trainable = False

  p,g = resnet_model(p),resnet_model(g)

  # Conv 3

  block3 = [
    layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
    layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
    layers.MaxPooling2D((2,2), strides=(2,2)),
  ]

  BlueUnit3 = tf.keras.Sequential(block3)

  p,g = BlueUnit3(p),BlueUnit3(g)

  # Conv4

  conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')
  p_bn1 = layers.BatchNormalization()
  g_bn1 = layers.BatchNormalization()

  p,g = conv1(p),conv1(g)
  p,g = p_bn1(p),g_bn1(g)

  shared_green = [
    layers.ReLU(),
    layers.MaxPooling2D(pool_size=pool_size_1, strides=strides),
  ]

  GreenUnit1 = tf.keras.Sequential(shared_green)

  p,g = GreenUnit1(p),GreenUnit1(g)

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
  g_se = SE1(g)
  p = layers.Multiply()([p, p_se])
  g = layers.Multiply()([g, g_se])

  # # Conv5

  # conv2 = layers.Conv2D(32, kernel_size=kernel_size_1, strides=strides, activation='relu')
  # p_bn2 = layers.BatchNormalization()
  # g_bn2 = layers.BatchNormalization()

  # p,g = conv2(p),conv2(g)
  # p,g = p_bn2(p),g_bn2(g)

  # GreenUnit2 = tf.keras.Sequential(shared_green)
  # p = GreenUnit2(p)
  # g = GreenUnit2(g)

  # SE2 = tf.keras.Sequential(shared_se)
  # p_se = SE2(p)
  # g_se = SE2(g)
  # p = layers.Multiply()([p, p_se])
  # g = layers.Multiply()([g, g_se])

  # Conv layers finished

  # Element-wise subtraction
  sub = layers.Subtract()([p,g])

  # Flatten
  x = layers.GlobalAveragePooling2D()(sub)

  # Dense layers
  x = layers.Dense(64, activation='sigmoid')(x)
  x = layers.Dense(64, activation='sigmoid')(x)
  output = layers.Dense(2, activation='softmax')(x)

  # Model
  model = models.Model([p_input,g_input],output)

  return model
