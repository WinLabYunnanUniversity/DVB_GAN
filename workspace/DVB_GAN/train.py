import time
import os
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.optimizers import RMSprop
import keras.backend as K
import model
K.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

latent_dim = 200
input_shape = (168, 224, 1)


def train_wgan(batch_size, epochs, image_shape):

    enc_model_1 = model.make_encoder()
    img = Input(shape=input_shape)
    z = enc_model_1(img)
    encoder1 = Model(img, z)

    z = Input(shape=(latent_dim,))
    modelG = model.construct_generator()
    gen_img = modelG(z)
    generator = Model(z, gen_img)
    critic = model.construct_critic(image_shape)

    critic.trainable = False
    img = Input(shape=input_shape)
    z = encoder1(img)

    img_ = generator(z)
    real = critic(img_)
    optimizer = RMSprop(0.0002)
    gan = Model(img, [real, img_])
    gan.compile(loss=[model.wasserstein_loss, 'mean_absolute_error'], optimizer=optimizer, metrics=None)

    X_train = model.load_data(168, 224)
    number_of_batches = int(X_train.shape[0] / batch_size)

    generator_iterations = 0
    d_loss = 0

    for epoch in range(epochs):

        current_batch = 0

        while current_batch < number_of_batches:

            start_time = time.time()
            # In the first 25 epochs, the critic is updated 100 times
            # for each generator update. In the other epochs the default value is 5
            if generator_iterations < 25 or (generator_iterations + 1) % 500 == 0:
                critic_iterations = 100
            else:
                critic_iterations = 5

            # Update the critic a number of critic iterations
            for critic_iteration in range(critic_iterations):

                if current_batch > number_of_batches:
                    break

                # real_images = dataset_generator.next()
                it_index = np.random.randint(0, number_of_batches - 1)
                real_images = X_train[it_index * batch_size:(it_index + 1) * batch_size]

                current_batch += 1

                # The last batch is smaller than the other ones, so we need to
                # take that into account
                current_batch_size = real_images.shape[0]
                # Generate images
                z = encoder1.predict(real_images)
                generated_images = generator.predict(z)
                # generated_images = generator.predict(noise)

                # Add some noise to the labels that will be fed to the critic
                real_y = np.ones(current_batch_size)
                fake_y = np.ones(current_batch_size) * -1
                # print('real_y', real_y)

                # Let's train the critic
                critic.trainable = True

                # Clip the weights to small numbers near zero
                for layer in critic.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(w, -0.01, 0.01) for w in weights]
                    layer.set_weights(weights)

                d_real = critic.train_on_batch(real_images, real_y)
                d_fake = critic.train_on_batch(generated_images, fake_y)

                d_loss = d_real - d_fake

            # Update the generator
            critic.trainable = False
            itt_index = np.random.randint(0, number_of_batches - 1)
            imgs = X_train[itt_index * batch_size:(itt_index + 1) * batch_size]
            # We try to mislead the critic by giving the opposite labels
            fake_yy = np.ones(current_batch_size)
            g_loss = gan.train_on_batch(imgs, [fake_yy, imgs])

            time_elapsed = time.time() - start_time
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_G_imgs: %f -> %f s'
                  % (epoch, epochs, current_batch, number_of_batches, generator_iterations,
                     d_loss, g_loss[0], g_loss[1], time_elapsed))

            generator_iterations += 1


if __name__ == '__main__':
    BS = 32
    image_shape = (168, 224, 1)
    Epochs = 1000
    train_wgan(BS, Epochs, image_shape)
