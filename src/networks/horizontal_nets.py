from keras.layers import Input, Lambda, Conv2D, Dense, Flatten, MaxPooling2D, SeparableConv2D
from keras.layers import BatchNormalization, Dropout, concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop, Nadam
import tensorflow as tf


class HorizontalNetworkV5():
    def __init__(self, input_shape, optimizer):
        self.input_shape = input_shape
        self.optimizer = optimizer

    def build_net(self, num_outputs):
        def horizontal_block(input, block_name):
            conv0 = Conv2D(8, (9, 9), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv0'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv0 = BatchNormalization(name='{}_conv0_batchnorm'.format(block_name))(conv0)

            conv1 = Conv2D(8, (7, 7), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv1'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv1 = BatchNormalization(name='{}_conv1_batchnorm'.format(block_name))(conv1)

            conv2 = Conv2D(8, (5, 5), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv2'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv2 = BatchNormalization(name='{}_conv2_batchnorm'.format(block_name))(conv2)

            conv3 = Conv2D(8, (3, 3), padding="same", activation='relu',
                           kernel_initializer="he_normal", name='{}_conv3'.format(block_name),
                           kernel_regularizer=l2(1e-2))(input)
            conv3 = BatchNormalization(name='{}_conv3_batchnorm'.format(block_name))(conv3)

            return concatenate([conv0, conv1, conv2, conv3])

        net_input = Input(shape=self.input_shape, name="input")

        convnet = horizontal_block(net_input, "block0")

        convnet = horizontal_block(convnet, "block1")

        convnet = horizontal_block(convnet, "block2")

        convnet = horizontal_block(convnet, "block3")

        convnet = horizontal_block(convnet, "block4")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block5")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block6")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = horizontal_block(convnet, "block7")
        convnet = MaxPooling2D(padding="valid")(convnet)

        convnet = Conv2D(512, (3, 3), padding="same", activation='relu',
                               kernel_initializer="he_normal", name='final_conv',
                               kernel_regularizer=l2(1e-2))(convnet)
        convnet = Flatten()(convnet)
        convnet = Dense(512, activation="relu", kernel_regularizer=l2(1e-3),
                              kernel_initializer="he_normal")(convnet)
        convnet = Dropout(0.5)(convnet)

        classif = Dense(num_outputs, activation='softmax',
                                     name="classification")(convnet)

        net = Model(inputs=[net_input],
                            outputs=[classif])
        net.compile(loss={"classification": "categorical_crossentropy"},
                            optimizer=self.optimizer,
                            metrics={"classification": "accuracy"})
        return net
