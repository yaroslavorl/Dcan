import numpy as np
import matplotlib.pyplot as plt


def read_img(path):

    with open(path, 'rb') as f:

        everything = np.fromfile(f, dtype=np.uint8)
        img = np.reshape(everything, (-1, 3, 96, 96))

    return np.transpose(img, (0, 3, 2, 1))


images = read_img('STL10/stl10_binary/unlabeled_X.bin')

print(images.shape)
print(images[0].shape)
n = 10
plt.figure(figsize=(20, 4))

for i in range(1, n + 1):

    ax = plt.subplot(2, n, i)
    plt.imshow(images[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()