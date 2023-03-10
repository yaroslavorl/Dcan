import matplotlib.pyplot as plt

from mxnet_net.set import data


def pictures(model):

    _, x_test = data()

    decoded_imgs = model(x_test.as_nd_ndarray().astype('float32'))

    n = 5
    plt.figure(figsize=(20, 4))

    for i in range(1, n + 1):

        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(8, 8).asnumpy())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(8, 8).asnumpy())
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
