from tensorflow.keras.datasets import cifar10,fashion_mnist,mnist
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os

import utils
BUFFER_SIZE = 1000
dim_mnist = 1000
def load_images_schifo(path, opt, test = False):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        seed=123,
        image_size=(opt["img_height"], opt["img_width"]),
        batch_size= opt["batch_size"] if not test else 3,
        color_mode = 'rgb'
        )

    return dataset

def decode_img(img, opt):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=opt["channels"])
  # resize the image to the desired size
  return tf.image.resize(img, [opt["img_height"], opt["img_width"]])

def process_path(file_path, opt):
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img, opt)
  return img

def load_images_path(path):
    dataset = tf.data.Dataset.list_files(path + "/*.png")
    return dataset

def get_emoji_dataset(emoji_type, opt):

    train_path = os.path.join("emojis", emoji_type)
    train_data_path = load_images_path(train_path)
    train_data = train_data_path.map(lambda x: process_path(x, opt))
    train_data = train_data.map(lambda x: utils.preprocess_image_train(x, opt)).cache().shuffle(BUFFER_SIZE).batch(opt["batch_size"])

    test_path = os.path.join("emojis", "Test_" + emoji_type)
    test_data_path = load_images_path(test_path)
    test_data = test_data_path.map(lambda x: process_path(x, opt))
    test_data = test_data.map(lambda x: utils.preprocess_image_test(x, opt)).cache().shuffle(
                                                                BUFFER_SIZE).batch(3)
    return train_data, test_data

def get_fashion_dataset(opt):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    np.random.shuffle(X_train.copy())
    np.random.shuffle(X_test.copy())
    X_train = X_train[:dim_mnist,:]
    X_test = X_test[:dim_mnist,:]

    X_train = np.expand_dims(X_train, axis=3)
    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    X_train = X_train.map(lambda x: utils.preprocess_image_train(x, opt)).cache().shuffle(BUFFER_SIZE).batch(opt["batch_size"])

    X_test = np.expand_dims(X_test, axis=3)
    X_test = tf.data.Dataset.from_tensor_slices(X_test)
    X_test = X_test.map(lambda x: utils.preprocess_image_test(x, opt)).cache().shuffle(BUFFER_SIZE).batch(3)

    return X_train, X_test

def get_digit_dataset(opt):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.expand_dims(X_train, axis=3)
    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    X_train = X_train.map(lambda x: utils.preprocess_image_train(x, opt)).cache().shuffle(BUFFER_SIZE).batch(opt["batch_size"])

    X_test = np.expand_dims(X_test, axis=3)
    X_test = tf.data.Dataset.from_tensor_slices(X_test)
    X_test = X_test.map(lambda x: utils.preprocess_image_test(x, opt)).cache().shuffle(BUFFER_SIZE).batch(3)

    return X_train, X_test

def get_horse2zebra(opt):
    dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                                  with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    train_horses = train_horses.map(lambda x, y: utils.preprocess_image_train(x, opt)).cache().shuffle(
        BUFFER_SIZE).batch(opt["batch_size"])

    train_zebras = train_zebras.map(lambda x, y: utils.preprocess_image_train(x, opt)).cache().shuffle(
        BUFFER_SIZE).batch(opt["batch_size"])

    test_horses = test_horses.map(
        lambda x, y: utils.preprocess_image_test(x, opt)).cache().shuffle(
        BUFFER_SIZE).batch(3)

    test_zebras = test_zebras.map(
        lambda x, y: utils.preprocess_image_test(x, opt)).cache().shuffle(
        BUFFER_SIZE).batch(3)
    return train_horses, train_zebras, test_horses, test_zebras