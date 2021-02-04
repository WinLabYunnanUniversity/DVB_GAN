import os, cv2
import numpy as np
from random import random
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape, MaxPooling2D
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal
import keras.backend as K
K.clear_session()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


latent_dim = 200
input_shape = (168, 224, 1)


def load_data(img_rows, img_cols):
    # img_rows, img_cols = 336, 456
    datPaths = []
    for root, dirs, files in os.walk(r"data/train", topdown=False):
        for name in files:
            if (name != ".DS_Store"):
                l = os.path.join(root, name)
                datPaths.append(l)
        for name in dirs:
            print(os.path.join(root, name))
    datPaths = sorted(datPaths)
    random.seed(2020)
    random.shuffle(datPaths)
    print(datPaths[0])
    #
    data = []
    for imagePath in datPaths:
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (img_cols, img_rows))
        data.append(image)

    data = np.array(data, dtype="float")
    print(data.shape)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    train_data = data
    print(train_data.shape[0], ' samples')
    # 测试数据
    X_train = train_data.astype('float32')
    print('X_train shape:', X_train.shape)
    print(X_train[0], 'train samples')
    X_train = (X_train - 127.5) / 127.5

    X_train = X_train.reshape(-1, img_rows, img_cols, 1)
    print(X_train.shape, 'X_train.shape')
    return X_train

# Let's define our Wasserstein Loss function. We apply the mean in order to be
# able to compare outputs with different batch sizes
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# Creates the critic model. This model tries to classify images as real
# or fake.
def construct_critic(image_shape):

    # weights need to be initialized with close values near zero to avoid
    # clipping
    weights_initializer = RandomNormal(mean=0., stddev=0.01)

    critic = Sequential()
    critic.add(Conv2D(filters=32, kernel_size=(7, 7),
                      strides=(2, 2), padding='same',
                      data_format='channels_last',
                      kernel_initializer=weights_initializer,
                      input_shape=(image_shape)))
    critic.add(LeakyReLU(0.2))
    critic.add(MaxPooling2D(pool_size=(2, 2)))

    critic.add(Conv2D(filters=64, kernel_size=(7, 7),
                      strides=(2, 2), padding='same',
                      data_format='channels_last',
                      kernel_initializer=weights_initializer))
    critic.add(BatchNormalization(momentum=0.5))
    critic.add(LeakyReLU(0.2))
    critic.add(MaxPooling2D(pool_size=(2, 2)))

    critic.add(Conv2D(filters=128, kernel_size=(7, 7),
                      strides=(2, 2), padding='same',
                      data_format='channels_last',
                      kernel_initializer=weights_initializer))
    critic.add(BatchNormalization(momentum=0.5))
    critic.add(LeakyReLU(0.2))
    critic.add(MaxPooling2D(pool_size=(2, 2)))
    critic.add(Flatten())

    # We output two layers, one witch predicts the class and other that
    # tries to figure if image is fake or not
    critic.add(Dense(units=1, activation=None))
    optimizer = RMSprop(0.0002)
    critic.compile(loss=wasserstein_loss,
                   optimizer=optimizer,
                   metrics=None)
    print(critic.summary())
    return critic


# Creates the generator model. This model has an input of random noise and
# generates an image that will try mislead the critic.
def make_encoder():
    modelE = Sequential()
    modelE.add(Conv2D(32, kernel_size=(7, 7), padding="same", input_shape=input_shape))
    modelE.add(BatchNormalization(momentum=0.5))
    modelE.add(Activation("relu"))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))
    modelE.add(Conv2D(64, kernel_size=(7, 7), padding="same"))
    modelE.add(BatchNormalization(momentum=0.5))
    modelE.add(Activation("relu"))
    modelE.add(MaxPooling2D(pool_size=(2, 2)))
    # modelE.add(Conv2D(64, kernel_size=(7, 7), padding="same"))
    # modelE.add(BatchNormalization(momentum=0.5))
    # modelE.add(Activation("relu"))
    # modelE.add(MaxPooling2D(pool_size=(2, 2)))
    modelE.add(Conv2D(128, kernel_size=(7, 7), padding="same"))
    # modelE.add(BatchNormalization(momentum=0.8))
    modelE.add(Activation("relu"))
    modelE.add(Flatten())
    modelE.add(Dense(latent_dim))
    print(modelE.summary())
    return modelE
# Important note: in the original pytorch implementation of the artice, the biases
# are set to false, here I left them as default.


def construct_generator():

    weights_initializer = RandomNormal(mean=0., stddev=0.01)

    generator = Sequential()

    generator.add(Dense(units=21 * 28 * 256,
                        kernel_initializer=weights_initializer,
                        input_dim=latent_dim))
    generator.add(Reshape(target_shape=(21, 28, 256)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=128, kernel_size=(7, 7),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=weights_initializer))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=64, kernel_size=(7, 7),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=weights_initializer))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))


    generator.add(Conv2DTranspose(filters=1, kernel_size=(7, 7),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=weights_initializer))
    generator.add(Activation('tanh'))

    optimizer = RMSprop(0.0002)
    generator.compile(loss=wasserstein_loss,
                      optimizer=optimizer,
                      metrics=None)
    return generator




