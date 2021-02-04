Structure of Project " Spectrum Monitoring of Radio Digital Video Broadcasting Based on an Improved GAN ":

## 1.Networks:

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
## 2.Dependencies
    python + keras + matlab(R2020a):
      numpy==1.19.2  scipy==1.5.3  matplotlib==2.2.3`
      Keras==2.2.4   matplotlib==3.3.2  
      scikit-learn==0.23.2  tensorflow-gpu==1.10.0
      tqdm==4.15.0  opencv-python==3.4.1.15
      

## 3.Dataset
simulated dataset：`platform:MATLAB R2020a`
   link:https://pan.baidu.com/s/163mROSdDrfjDI6xLlZTzAg  19uy 

   == Run  _commdvbt_  in command window, and receive signal as S 
   
   == Process original data and extract spectrum:
   
     signal = S.signals.values;
     [s1 s2 s3]=size(signal);
     signal = reshape(signal, s1, s3);
     Fs = 9.14*1000000;
     spec = [];
     savename = ['data'];
     for i=1:s3
         spectrum_scope = dsp.SpectrumAnalyzer('SampleRate', Fs);
         spectrum_scope(signal(:,i));
         release(spectrum_scope);
         datai = getSpectrumData(spectrum_scope);
         y = cell2mat(datai.Spectrum);
         figure('visible','off')
         p=plot(y,'k');
         axis off;
         color_savename=[num2str(i),'.png'];
         set(gca,'xtick',[],'ytick',[],'xcolor','w','ycolor','w','box','off') 
         saveas(p,color_savename);
         img=imread(color_savename);
         img_shape = size(img);
         up =45;
         down = img_shape(1)-110;
         left = 120;
         right = img_shape(2)-100;
         clip = img(up:down,left:right);
         I_img = imcomplement(clip);%反转为黑底白线
         I_img = imresize(I_img, [168, 224]);
         imwrite(I_img,color_savename); %将灰度图片写入
         spec(i,:)=y';
     end
     saldir = 'data_normal\';
     savePath = [saldir savename '.mat'];
     save(savePath,'spec'); 
     

true dataset：
   link:https://pan.baidu.com/s/1d3l2XBuVFSb1nO3mTbnnQg  d8dq
   
## 4.Training
   
    BS = 32
    image_shape = (168, 224, 1)
    Epochs = 1000
    optimizer = RMSprop(0.0002)
    generator.compile(loss=wasserstein_loss,
                      optimizer=optimizer,
                      metrics=None)optimizer = RMSprop(0.0002)
    generator.compile(loss=wasserstein_loss,
                      optimizer=optimizer,
                      metrics=None)
 

## 5.Evaluate
    def AccCount(a, b):
      normal_num = 0
      for i in a:
          if i < b:   
              normal_num += 1
      normal = normal_num/len(a)
      abnormal_num = 0
      for i in a:
          if i >= b:  
              abnormal_num += 1
      abnormal = abnormal_num/len(a)
      return normal_num, normal, abnormal_num, abnormal
      
     encoder = load_model(r'results\enc1_n.h5')
     generator = load_model(r'results\gen_n.h5')
     input_arr = X_test_original
     z_gen_ema = encoder1.predict(input_arr)  # latent code
     reconstruct_ema = generator.predict(z_gen_ema)  # reconstructed images  
     reconstruct_ema = reconstruct_ema.reshape(-1, img_rows, img_cols, 1)   
     residual = reconstruct_ema - X_test_original  
    
     threshold = []
     for i in range(len(residual)):
         th = np.mean(np.abs(residual[i]))
         threshold.append(th)
     num = int(len(residual)/2)
     threshold = np.array(threshold)

    # principle：3sigmoid
    arr_mean = np.mean(threshold[:num])
    arr_std = np.std(threshold[:num], ddof=1)
    T = arr_mean + 3 * arr_std
    print(AccCount(threshold[:num], T))




