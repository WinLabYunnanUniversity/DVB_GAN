from __future__ import print_function
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
K.clear_session()

img_rows, img_cols = 168, 224


def AccCount(a, b):
    normal_num = 0
    for i in a:
        if i < b:   # 可依需要修改条件
            normal_num += 1
    normal = normal_num/len(a)
    abnormal_num = 0
    for i in a:
        if i >= b:  # 可依需要修改条件
            abnormal_num += 1
    abnormal = abnormal_num/len(a)
    return normal_num, normal, abnormal_num, abnormal


def get_dataset(img_path):
    datPaths = []
    data = []
    for root, dirs, files in os.walk(img_path, topdown=False):
        for name in files:
            if (name != ".DS_Store"):
                imagePath = os.path.join(root, name)
                datPaths.append(imagePath)
                image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (img_cols, img_rows))
                data.append(image)
    data = np.array(data, dtype="float")
    # print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
    data = data.astype('float32')
    data = (data - 127.5) / 127.5
    data = data.reshape(-1, img_rows, img_cols, 1)
    # data = data[:10]
    return data


if __name__ == '__main__':
    X_train =get_dataset('train_path')
    X_test = get_dataset('test_path')
    X_test_original = np.concatenate((X_train, X_test), axis=0)
    print('X_train.shape=', X_train.shape, '\nX_test.shape=',
          X_test.shape, '\nX_test_original.shape=', X_test_original.shape)

    encoder1 = load_model(r'results\enc1_n.h5')
    generator = load_model(r'results\gen_n.h5')
    input_arr = X_test_original
    z_gen_ema = encoder1.predict(input_arr)  # latent code  (1000, 200)
    reconstruct_ema = generator.predict(z_gen_ema)  # reconstruct images  (1000, 168, 224, 1)
    reconstruct_ema = reconstruct_ema.reshape(-1, img_rows, img_cols, 1)   # (1000, 168, 224, 1)
    residual = reconstruct_ema - X_test_original  # residual value (1000, 168, 224, 1)

    # Threshold
    threshold = []
    for i in range(len(residual)):
        count = np.mean(np.abs(residual[i]))
        threshold.append(count)

    num = X_train.shape[0]
    threshold = np.array(threshold)  # (1000, )

    plt.figure()
    plt.hist(threshold[:num], bins=20, facecolor='red', alpha=0.5, label='Normal')
    plt.hist(threshold[num:], bins=20, facecolor='blue', alpha=0.5, label='Abnormal')
    plt.legend(fontsize=16), plt.xticks(fontsize=16), plt.yticks(fontsize=18), plt.show()

    # principle：3sigmoid
    arr_mean = np.mean(threshold[:num])
    arr_std = np.std(threshold[:num], ddof=1)
    T = arr_mean + 3 * arr_std
    print(AccCount(threshold[:num], T))
