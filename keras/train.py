from model_keras import dcan
from test_weights import data

patterns = 30
bath_size = 32
epochs = 25

model = dcan(patterns)
model.compile(loss='mean_squared_error',  optimizer='adam')

x_train, x_test = data()

model.fit(x_train, x_train,
          batch_size=bath_size,
          epochs=epochs,
          validation_data=(x_test, x_test))

model.save('dcan')