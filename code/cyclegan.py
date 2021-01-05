
import tensorflow as tf
import tensorflow_datasets as tfds

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

import model
import dataloader
import utils

tfds.disable_progress_bar()



def main(opt):


    # Create sample and checkpoint directories
    if opt["use_cycle_consistency_loss"]:
        os.makedirs("images/{}".format(opt["dataset_name"]), exist_ok=True)
        os.makedirs("plots/{}".format(opt["dataset_name"]), exist_ok=True)
    else:
        os.makedirs("no_cycle/images/{}".format(opt["dataset_name"]), exist_ok=True)
        os.makedirs("no_cycle/plots/{}".format(opt["dataset_name"]), exist_ok=True)

    # Create train and test dataloaders for images from the two domains X and Y
    if opt["dataset_name"] == "fashion2digit":
        dataloader_X, test_dataloader_X = dataloader.get_fashion_dataset(opt)
        dataloader_Y, test_dataloader_Y = dataloader.get_digit_dataset(opt)

    if opt["dataset_name"] == "apple2windows":
        dataloader_X, test_dataloader_X = dataloader.get_emoji_dataset(opt["X"], opt)
        dataloader_Y, test_dataloader_Y = dataloader.get_emoji_dataset(opt["Y"], opt)

    if opt["dataset_name"] == "horse2zebra":
        dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y = dataloader.get_horse2zebra(opt)
    print("data loaded...")

    return train(opt, dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y)

def train(opt, dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y):

    input_shape = (None, None, opt["channels"])

    norm_type = 'batchnorm'
    if opt["batch_size"] == 1:
        norm_type = 'instancenorm'

    # models
    G_YtoX = model.Generator(input_shape, opt, norm_type)
    G_XtoY = model.Generator(input_shape, opt, norm_type)
    D_X = model.Discriminator(input_shape, norm_type)
    D_Y = model.Discriminator(input_shape, norm_type)

    # summary
    G_YtoX.summary()
    D_X.summary()

    # optimizers
    G_YtoX_optimizer = tf.optimizers.Adam(opt["lr_g"], beta_1=opt["b1"])
    G_XtoY_optimizer = tf.optimizers.Adam(opt["lr_g"], beta_1=opt["b1"])

    D_X_optimizer = tf.optimizers.Adam(opt["lr_d"], beta_1=opt["b1"])
    D_Y_optimizer = tf.optimizers.Adam(opt["lr_d"], beta_1=opt["b1"])

    # set checkpoint
    ckpt, ckpt_manager = model.set_checkpoint(opt, G_YtoX, G_XtoY, D_X, D_Y, G_YtoX_optimizer, G_XtoY_optimizer, D_X_optimizer, D_Y_optimizer)

    test_X = next(iter(test_dataloader_X))
    test_Y = next(iter(test_dataloader_Y))

    iterations = 0
    total_disc_loss_list, total_gen_loss_list, test_loss_list = utils.load_losses_list(opt)
    mean_cycle_loss_test = 0
    count = 0

    for epoch in range(opt["epoch"], opt["n_epochs"] + 1):
        start = time.time()

        for image_x, image_y in tf.data.Dataset.zip((dataloader_X, dataloader_Y)):

            total_disc_loss, total_gen_loss = train_step(image_x, image_y, G_YtoX, G_XtoY, D_X, D_Y, G_YtoX_optimizer, G_XtoY_optimizer, D_X_optimizer, D_Y_optimizer, opt)
            total_disc_loss_list.append(total_disc_loss)
            total_gen_loss_list.append(total_gen_loss)
            if iterations % 200 == 0:
                print('iteration:{}'.format(iterations))
                # Using a consistent image so that the progress of the model is visible.
                cycle_loss_test = utils.save_samples(epoch, test_Y, test_X, G_YtoX, G_XtoY, opt)
                mean_cycle_loss_test  += cycle_loss_test
                count +=1
                utils.plot_losses(total_disc_loss_list, total_gen_loss_list, opt, epoch, "Epoch:{}".format(epoch))
                
            iterations += 1

        clear_output(wait=True)
        # Using a consistent image so that the progress of the model is visible. Only one plot is keept per epoch
        cycle_loss_test = utils.save_samples(epoch, test_Y, test_X, G_YtoX, G_XtoY, opt)
        mean_cycle_loss_test += cycle_loss_test
        count += 1
        test_loss_list.append(mean_cycle_loss_test / count)
        mean_cycle_loss_test=0
        count = 0

        utils.plot_losses(total_disc_loss_list, total_gen_loss_list, opt, epoch, "Epoch:{}".format(epoch))
        utils.plot_test_loss(test_loss_list, opt, epoch)
        utils.save_losses_list(total_disc_loss_list, total_gen_loss_list, test_loss_list, opt)
        
        print("Sample saved...")
        if (epoch + 1) % opt["checkpoint_interval"] == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch ,
                                                                ckpt_save_path))

        print('Time taken for epoch {} is {} sec\n'.format(epoch,
                                                           time.time() - start))
    return G_YtoX, G_XtoY, D_X, D_Y

#@tf.function
def train_step(real_x, real_y, G_YtoX, G_XtoY, D_X, D_Y, G_YtoX_optimizer, G_XtoY_optimizer, D_X_optimizer, D_Y_optimizer, opt):
    # persistent is set to True because the tape is used more than once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G_XtoY translates X -> Y
        # Generator G_YtoX translates Y -> X.

        fake_y = G_XtoY(real_x, training=True)
        cycled_x = G_YtoX(fake_y, training=True)

        fake_x = G_YtoX(real_y, training=True)
        cycled_y = G_XtoY(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = G_XtoY(real_x, training=True)
        same_y = G_XtoY(real_y, training=True)

        disc_real_x = D_X(real_x, training=True)
        disc_real_y = D_Y(real_y, training=True)

        disc_fake_x = D_X(fake_x, training=True)
        disc_fake_y = D_Y(fake_y, training=True)

        # calculate the loss
        G_XtoY_loss = model.generator_loss(disc_fake_y)
        G_YtoX_loss = model.generator_loss(disc_fake_x)

        if opt["use_cycle_consistency_loss"]:
            total_cycle_loss = model.calc_cycle_loss(real_x, cycled_x) + model.calc_cycle_loss(real_y, cycled_y)
        else:
            total_cycle_loss = 0

        # Total generator loss = adversarial loss + cycle loss
        total_G_XtoY_loss = G_XtoY_loss + total_cycle_loss + model.identity_loss(real_y, same_y)
        total_G_YtoX_loss = G_YtoX_loss + total_cycle_loss + model.identity_loss(real_x, same_x)

        disc_x_loss, update_D_X = model.discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss, update_D_Y = model.discriminator_loss(disc_real_y, disc_fake_y)

        # total loss to be shown
        total_disc_loss = (disc_x_loss + disc_y_loss)/2
        total_gen_loss = (total_G_XtoY_loss + total_G_YtoX_loss) / 2

    # Calculate the gradients for generator and discriminator

    G_XtoY_gradients = tape.gradient(total_G_XtoY_loss,
                                          G_XtoY.trainable_variables)
    G_YtoX_gradients = tape.gradient(total_G_YtoX_loss,
                                          G_YtoX.trainable_variables)
    if update_D_X:
        D_X_gradients = tape.gradient(disc_x_loss,
                                                D_X.trainable_variables)
    if update_D_Y:
        D_Y_gradients = tape.gradient(disc_y_loss,
                                              D_Y.trainable_variables)

    # Apply the gradients to the optimizer

    G_XtoY_optimizer.apply_gradients(zip(G_XtoY_gradients,
                                              G_XtoY.trainable_variables))

    G_YtoX_optimizer.apply_gradients(zip(G_YtoX_gradients,
                                              G_YtoX.trainable_variables))
    if update_D_X:
        D_X_optimizer.apply_gradients(zip(D_X_gradients,
                                                    D_X.trainable_variables))
    if update_D_Y:
        D_Y_optimizer.apply_gradients(zip(D_Y_gradients,
                                                    D_Y.trainable_variables))

    return total_disc_loss, total_gen_loss

if __name__ == "__main__":
    main()