from mxnet.gluon import nn
from mxnet.ndarray import reshape


def dcan(patterns):

    model = nn.Sequential()

    model.add(

        nn.Flatten(),
        nn.BatchNorm(),
        nn.Dense(patterns),
        nn.BatchNorm(),
        nn.Dense(64),
        nn.Lambda(lambda x: reshape(x, (-1, 1, 8, 8))),

        nn.Conv2D(64, (9, 9), activation='relu', padding=4),
        nn.Conv2D(32, (1, 1), activation='relu', padding=0),
        nn.Conv2D(1, (5, 5), activation='relu', padding=2)

    )

    return model
