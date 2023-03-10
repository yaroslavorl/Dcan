from mxnet import autograd
from mxnet import gluon

from model_mxnet import dcan
from set import data
from test import pictures


def fit_model(model, train_set, epochs, loss_func=gluon.loss.L2Loss()):

    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.01})

    for epoch in range(1, epochs + 1):
        for batch in train_set:

            with autograd.record():

                prediction = model(batch.as_nd_ndarray().astype('float32'))
                loss = loss_func(prediction, batch.as_nd_ndarray().astype('float32'))

            loss.backward()
            trainer.step(len(batch))

        loss = loss.mean()
        print(f'epoch : {epoch} loss : {loss.asscalar()}')


model = dcan(patterns=20)
model.initialize()

x_train, x_test = data(bath_size=33)

fit_model(model, x_train, epochs=15)
#model.save_parameters('params_file.params')

pictures(model)





