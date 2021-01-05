#import tensorflow as tf
from tensorflow import ones_like, zeros_like, reduce_mean, abs, random_normal_initializer, math, nn
from tensorflow.keras.layers import Layer
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.keras import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization, ReLU, Dropout, Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten, ZeroPadding2D
from keras.optimizers import Adam
from keras import initializers, regularizers, constraints
import os

LAMBDA = 10
smooth = 0.0

loss_obj = BinaryCrossentropy(from_logits=True)
#loss_obj = tf.keras.losses.MeanSquaredError()
#acc = tf.keras.metrics.Accuracy()

def set_checkpoint(opt, G_YtoX, G_XtoY, D_X, D_Y, G_YtoX_optimizer, G_XtoY_optimizer, D_X_optimizer, D_Y_optimizer):
  if opt["use_cycle_consistency_loss"]:
    os.makedirs("./checkpoints/{}/train".format(opt["dataset_name"]), exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", opt["dataset_name"],"train")
  else:
    os.makedirs("./no_cycle/checkpoints/{}/train".format(opt["dataset_name"]), exist_ok=True)
    checkpoint_path = os.path.join("no_cycle","checkpoints", opt["dataset_name"], "train")

  ckpt = Checkpoint(G_YtoX_optimizer=G_YtoX_optimizer,
                             G_XtoY_optimizer=G_XtoY_optimizer,
                             D_X_optimizer=D_X_optimizer,
                             D_Y_optimizer=D_Y_optimizer,
                             G_YtoX=G_YtoX,
                             G_XtoY=G_XtoY,
                             D_X=D_X,
                             D_Y=D_Y
                             )

  ckpt_manager = CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

  return ckpt, ckpt_manager


def discriminator_loss(real, generated):
  update = True
  real_loss = loss_obj(ones_like(real) * (1 - smooth), real)

  generated_loss = loss_obj(zeros_like(generated) , generated)
  total_disc_loss = (real_loss + generated_loss) * 0.5

  return total_disc_loss, update

def generator_loss(generated):
  return loss_obj(ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = reduce_mean(abs(real_image - cycled_image))

    return LAMBDA * loss1

def cycle_loss(real_image, cycled_image):
    loss1 = reduce_mean(abs(real_image - cycled_image))

    return loss1

def identity_loss(real_image, same_image):
  loss = reduce_mean(abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

class InstanceNormalization(Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = nn.moments(x, axes=[1, 2], keepdims=True)
    inv = math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def downsample(filters, size, norm_type = 'batchnorm', apply_batchnorm=True):
  initializer = random_normal_initializer(0., 0.02)

  result = Sequential()
  result.add(
      Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
      if norm_type == 'batchnorm':
          result.add(BatchNormalization())
      else:
          result.add(InstanceNormalization())

  result.add(LeakyReLU())

  return result

def upsample(filters, size, norm_type = 'batchnorm', apply_dropout=False):
  initializer = random_normal_initializer(0., 0.02)

  result = Sequential()
  result.add(
    Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  if norm_type == 'batchnorm':
      result.add(BatchNormalization())
  else:
      result.add(InstanceNormalization())

  if apply_dropout:
      result.add(Dropout(0.5))

  result.add(ReLU())

  return result

def ResNet(filters, norm_type = 'batchnorm', apply_dropout=False):
    initializer = random_normal_initializer(0., 0.02)

    result = Sequential()
    result.add(
        Conv2D(filters, 3, strides=1, padding='same',
               kernel_initializer=initializer, use_bias=False))

    if norm_type == 'batchnorm':
        result.add(BatchNormalization())
    else:
        result.add(InstanceNormalization())

    result.add(LeakyReLU())

    return result



def Generator(input_shape, opt, norm_type = 'batchnorm'):
    inputs = Input(shape=input_shape)

    down_stack = [
        downsample(64, 4, norm_type , apply_batchnorm=False),
        downsample(128, 4, norm_type ),
        downsample(256, 4, norm_type),
        downsample(512, 4, norm_type),
        downsample(512, 4, norm_type),

    ]

    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True ),
        upsample(256, 4, norm_type, apply_dropout=True),
        upsample(128, 4, norm_type),
        upsample(64, 4, norm_type )
    ]

    initializer = random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(input_shape[2], 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)

def Discriminator(input_shape, norm_type = 'batchnorm'):
  initializer = random_normal_initializer(0., 0.02)

  inp = Input(shape=input_shape, name='input_image')

  x = inp

  down1 = downsample(64, 4, norm_type, False)(x)
  down2 = downsample(128, 4, norm_type)(down1)

  zero_pad1 = ZeroPadding2D()(down2) # (bs, 34, 34, 256)
  conv = Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

  if norm_type.lower() == 'batchnorm':
      batchnorm1 = BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
      batchnorm1 = InstanceNormalization()(conv)

  leaky_relu = LeakyReLU()(batchnorm1)

  zero_pad2 = ZeroPadding2D()(leaky_relu)

  last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

  return Model(inputs=inp, outputs=last)