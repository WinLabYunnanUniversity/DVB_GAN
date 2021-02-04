**Structure of Project " Spectrum Monitoring of Radio Digital 
Video Broadcasting Based on an Improved GAN ":

1.Model networks:

****
    =================== Critic network ===================
    def construct_critic(image_shape):
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
      critic.add(Dense(units=1, activation=None))
      optimizer = RMSprop(0.0002)
      critic.compile(loss=wasserstein_loss,
                   optimizer=optimizer,
                   metrics=None)
      return critic
      
    =================== Encoder network ===================  
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
      modelE.add(Conv2D(128, kernel_size=(7, 7), padding="same"))
      modelE.add(Activation("relu"))
      modelE.add(Flatten())
      modelE.add(Dense(latent_dim))
      print(modelE.summary())
      return modelE
        
    =================== Generator network ===================
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
      
    ========================================================================




