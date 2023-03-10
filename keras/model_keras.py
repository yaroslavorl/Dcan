from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Reshape



def dcan(patterns):

    model = Sequential()

    # encoder layers
    model.add(Conv2D(patterns, (28, 28)))
    model.add(BatchNormalization())

    # decoder layers
    model.add(Conv2DTranspose(patterns, (28, 28)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(1, (5, 5), activation='relu', padding='same'))

    return model
