import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os
import cv2


def plot_img(plot_data, k, savename):
    plt.plot(plot_data, 'k')
    plt.ylim((plot_data.min(), plot_data.max()))
    plt.xticks([]), plt.yticks([]), plt.axis('off')
    savename = savename + r"\%d.png" % k
    plt.savefig(savename)
    plt.close()

    # exit()

    plt.figure(figsize=(3, 3))
    im = cv2.imread(savename)
    # print("im=", im)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # exit()
    img_shape = gray.shape
    up = 55
    down = img_shape[0] - 40
    left = 90
    right = img_shape[1] - 80
    cropped = gray[up:down, left:right]
    image = cv2.resize(cropped, (224, 168))
    dst = 255-image
    cv2.imwrite(savename, dst)
    plt.close()


def gene_ab(test_path, bandwidth, power, savename):
    data = np.load(test_path)
    # data = data[:800]
    ab_a = 800
    ab_b = ab_a + bandwidth
    index = np.arange(ab_a, ab_b)
    # index = np.random.randint(0, 1536, size=bandwidth)
    print('###############', index)
    for i in range(1, data.shape[0]):
        plot_data1 = data[i, :]
        modify = plot_data1[index] + power
        plot_data2 = plot_data1
        plot_data2[index] = modify
        plot_img(plot_data2, i, savename)


def get_dataset(data_path):
    datPaths = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            if (name != ".DS_Store"):
                imagePath = os.path.join(root, name)
                datPaths.append(imagePath)
    return datPaths


def check(path):
    p = get_dataset(path)
    i = 0
    for pa in p:
        im = cv2.imread(pa)
        m = np.mean(im)
        if m < 5:
            i = i + 1
            print(pa)
    print('need to delete：', i)


if __name__ == '__main__':
    jj = 1
    Bandwidth = [20, 40, 70, 100, 200]
    Power = [3, 5, 8, 10, 15, 20]
    for b in range(len(Bandwidth)):
        bandwidth = Bandwidth[b]
        test_path = r'data.npy'
        for p in range(len(Power)):
            power = Power[p]
            save_path = r'abnorma\%d_%d' % (bandwidth, power)
            print('bandwidth = ', bandwidth, 'power = ', power)
            gene_ab(test_path, bandwidth, power, save_path)
            check(save_path)


