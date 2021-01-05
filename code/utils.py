import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from model import cycle_loss

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_crop(image, opt):
  cropped_image = tf.image.random_crop(image, size=[opt["img_height"], opt["img_width"], opt["channels"]])

  return cropped_image

def random_jitter(image, opt):
  # resizing
  image = tf.image.resize(image, [opt["img_height"] + int(opt["img_height"] * 0.2), opt["img_width"] + int(opt["img_width"] * 0.2)],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping
  image = random_crop(image, opt)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, opt):
  # resizing
  image = tf.image.resize(image, [opt["img_height"] ,
                                    opt["img_width"] ],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = random_jitter(image,opt)
  image = normalize(image)
  return image

def preprocess_image_test(image, opt):
  # resizing
  image = tf.image.resize(image, [opt["img_height"],
                                    opt["img_width"]],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = normalize(image)
  return image

def save_losses_list(total_disc_loss_list, total_gen_loss_list, test_loss_list, opt):
    if opt["use_cycle_consistency_loss"]:
        path = os.path.join('plots', opt["dataset_name"])
    else:
        path = os.path.join("no_cycle", 'plots', opt["dataset_name"])

    with open(os.path.join(path, "disc_loss"), 'wb') as f:
        pickle.dump(total_disc_loss_list, f)

    with open(os.path.join(path, "gen_loss"), 'wb') as f:
        pickle.dump(total_gen_loss_list, f)

    with open(os.path.join(path, "test_loss"), 'wb') as f:
        pickle.dump(test_loss_list, f)
        
def load_losses_list(opt):
    total_disc_loss_list, total_gen_loss_list, test_loss_list = [], [], []

    if opt["epoch"] == 0:
        return total_disc_loss_list, total_gen_loss_list, test_loss_list

    if opt["use_cycle_consistency_loss"]:
        path = os.path.join('plots', opt["dataset_name"])
    else:
        path = os.path.join("no_cycle",'plots', opt["dataset_name"])

    with open(os.path.join(path, "disc_loss"), 'rb') as f:
        total_disc_loss_list = pickle.load(f)

    with open(os.path.join(path, "gen_loss"), 'rb') as f:
        total_gen_loss_list = pickle.load(f)

    with open(os.path.join(path, "test_loss"), 'rb') as f:
        test_loss_list = pickle.load(f)

    return total_disc_loss_list, total_gen_loss_list, test_loss_list

def create_dict_param(epoch=0, n_epochs=200, img_height = 32, img_width = 32, num_channels = 3, use_cycle_consistency_loss = True, batch_size = 16, lr_d = 2e-5, lr_g = 2e-4, beta1 = 0.5, beta2 = 0.999, X = "Apple", Y = "Windows",
                dataset_name = "apple2windows", checkpoint_interval = 1, n_residual_blocks= 9, lambda_cyc =10):

    param = { "epoch" : epoch, # epoch to start training from
              "n_epochs" : n_epochs, # number of epochs of training
              "img_height" : img_height, # image height
              "img_width": img_width,  # image height
              "channels" : num_channels, # number of image channels
              "use_cycle_consistency_loss" : use_cycle_consistency_loss, # whether to include the cycle consistency term in the loss.
              "batch_size": batch_size , # The number of images in a batch.
              "lr_d": lr_d, # The learning rate discriminator
              "lr_g": lr_g, # The learning rate generator
              "b1": beta1, # parameter of the optimizer
              "b2": beta2, # parameter of the optimizer,
              "X" : X, # Choose the type of images for domain X.
              "Y" : Y, # Choose the type of images for domain Y.
              "dataset_name" : dataset_name,
              "checkpoint_interval" : checkpoint_interval, # interval between saving model checkpoints
              "n_residual_blocks" : n_residual_blocks, # number of residual blocks in generator
              "lambda_cyc" : lambda_cyc # cycle loss weight
              }

    return  param


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Parameters'.center(80))
    print('-' * 80)
    for key in opts:
        if opts[key]:
            print('{:>30}: {:<30}'.format(key, opts[key]).center(80))
    print('=' * 80)


def plot_losses(d_loss_list, g_loss_list, opt, iteration, title):
    if opt["use_cycle_consistency_loss"]:
        path = os.path.join('plots', opt["dataset_name"],'{}.png'.format(iteration))
    else:
        path = os.path.join("no_cycle",'plots', opt["dataset_name"], '{}.png'.format(iteration))

    plt.close()
    plt.plot(d_loss_list)
    plt.plot(g_loss_list)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.legend(['Discriminators', 'Generators'], loc='upper right')
    plt.savefig(path)
    plt.show()


def plot_test_loss(cycle_test_loss, opt, iteration):
    if opt["use_cycle_consistency_loss"]:
        path = os.path.join('plots', opt["dataset_name"], '{}_test.png'.format(iteration))
    else:
        path = os.path.join("no_cycle", 'plots', opt["dataset_name"], '{}_test.png'.format(iteration))

    plt.close()
    plt.plot(cycle_test_loss)
    plt.title("Reconstruction loss on a Test set")
    plt.ylabel('Cycle Loss')
    plt.xlabel('Epoch')
    plt.legend(['Loss'], loc='upper right')
    plt.savefig(path)
    plt.show()

def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
  """Saves samples from both generators X->Y and Y->X.
  """
  fake_X = G_YtoX(fixed_Y)  # Y -> X
  fake_Y = G_XtoY(fixed_X)  # X -> Y

  rec_X = G_YtoX(fake_Y)  # reconstructed X ... X -> Y -> X
  rec_Y = G_XtoY(fake_X)  # reconstructed Y ... Y -> X -> Y

  X, fake_X, rec_X = fixed_X, fake_X, rec_X
  Y, fake_Y, rec_Y = fixed_Y, fake_Y, rec_Y

  rgb = False
  if opts["channels"] == 3:
    rgb = True

  if opts["use_cycle_consistency_loss"]:
    path = os.path.join('images', opts["dataset_name"], 'sample-{:06d}-{}-{}-{}.png'.format(iteration, opts["X"], opts["Y"], opts["X"]))
  else:
    path = os.path.join("no_cycle",'images', opts["dataset_name"], 'sample-{:06d}-{}-{}-{}.png'.format(iteration, opts["X"], opts["Y"], opts["X"]))
  loss_X = plot_images(X, fake_Y, rec_X, path,"Epoch:{:06d} {}->{}->{}".format(iteration, opts["X"], opts["Y"], opts["X"]), rgb)

  if opts["use_cycle_consistency_loss"]:
    path = os.path.join('images', opts["dataset_name"],'sample-{:06d}-{}-{}-{}.png'.format(iteration, opts["Y"], opts["X"], opts["Y"]))
  else:
    path = os.path.join("no_cycle",'images', opts["dataset_name"],'sample-{:06d}-{}-{}-{}.png'.format(iteration, opts["Y"], opts["X"], opts["Y"]))

  loss_Y = plot_images(Y, fake_X, rec_Y, path,"Epoch:{:06d} {}->{}->{}".format(iteration, opts["Y"], opts["X"], opts["Y"]), rgb)
  return (loss_X + loss_Y) * 0.5

def plot_images(sources, targets, reconstructed, path, title, rgb):
  """Creates a grid consisting of pairs of columns, where the first column in
  each pair contains images source images and the second column in each pair
  contains images generated by the CycleGAN from the corresponding images in
  the first column.
  """

  loss = np.mean(cycle_loss(sources,reconstructed))
  title += " Cycle loss:{}".format(loss)

  # at most only 2 images are shown
  max_len = 3 if len(sources) > 3 else len(sources)
  
  plt.close()
  fig, axs = plt.subplots(max_len, 3,figsize=(25, 20))
  fig.suptitle(title, fontsize = 25)

  images = []
  for idx, (s, t, r) in enumerate(zip(sources, targets, reconstructed)):
    if idx >= 3:
      break


    s = s.numpy()
    t = t.numpy()
    r = r.numpy()

    if rgb:

      images.append(axs[idx, 0].imshow(((s + 1) * 127).astype(np.uint8)))
      images.append(axs[idx, 1].imshow(((t + 1) * 127).astype(np.uint8)))
      images.append(axs[idx, 2].imshow(((r + 1) * 127).astype(np.uint8)))
    else:

      s = np.squeeze(s, axis=-1)  # only the first two dimension, no channel
      t = np.squeeze(t, axis=-1)
      r = np.squeeze(r, axis=-1)
      images.append(axs[idx, 0].imshow(((s[:,:] + 1) * 127).astype(np.uint8), cmap='gray'))
      images.append(axs[idx, 1].imshow(((t[:,:] + 1) * 127).astype(np.uint8), cmap='gray'))
      images.append(axs[idx, 2].imshow(((r[:,:] + 1) * 127).astype(np.uint8), cmap='gray'))

    axs[idx, 0].axis('off')
    axs[idx, 1].axis('off')
    axs[idx, 2].axis('off')

  fig.savefig(path)
  return loss