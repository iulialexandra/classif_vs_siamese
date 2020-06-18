from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Input, Dropout, \
    concatenate, Lambda
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from tools.modified_sgd import Modified_SGD


class OriginalNetworkV2:
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape
        self.optimizer = optimizer


    def build_net(self, num_outputs):
        net_input = Input(self.input_shape, name="input")
        convnet = Conv2D(filters=64, kernel_size=(7, 7),
                         activation='relu',
                         padding="same",
                         kernel_regularizer=l2(1e-3),
                         name='Conv1')(net_input)
        convnet = MaxPooling2D(padding="same")(convnet)
        convnet = Dropout(0.5)(convnet)

        convnet = Conv2D(filters=128, kernel_size=(5, 5),
                         activation='relu',
                         padding="same",
                         kernel_regularizer=l2(1e-3),
                         name='Conv2')(convnet)
        convnet = MaxPooling2D(padding="same")(convnet)
        convnet = Dropout(0.5)(convnet)

        convnet = Conv2D(filters=128, kernel_size=(3, 3),
                         activation='relu',
                         padding="same",
                         kernel_regularizer=l2(1e-3),
                         name='Conv3')(convnet)
        convnet = MaxPool2D(padding="same")(convnet)
        convnet = Dropout(0.5)(convnet)

        convnet = Conv2D(filters=256, kernel_size=(3, 3),
                         activation='relu',
                         padding="same",
                         kernel_regularizer=l2(1e-3),
                         name='Conv4')(convnet)
        convnet = MaxPool2D(padding="same")(convnet)
        convnet = Dropout(0.5)(convnet)
        convnet = Flatten()(convnet)
        convnet = Dense(4096, activation="relu", kernel_regularizer=l2(1e-3),
                        kernel_initializer="he_normal")(convnet)
        classif = Dense(num_outputs, activation='softmax',
                                     name="classification")(convnet)

        net = Model(inputs=[net_input],
                            outputs=[classif])
        net.compile(loss={"classification": "categorical_crossentropy"},
                            optimizer=self.optimizer,
                            metrics={"classification": "accuracy"})
        return net
