from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization, PReLU, Dropout, MaxPooling2D
from keras.optimizers import Adam

import keras

import skimage.io as io
import numpy as np

audio_shape = (513, 25, 1)
dropout_rate = 0.2
bn_momentum = 0.8

key_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
           'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
           'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7',
           '8', '9', '0', ' ', ',', '.']

filter_num = [16, 32, 64, 128]

# classifies an audio shape into a key
class TapModel(object):
    def __init__(self, shape=audio_shape, weights_path=None):
        self.shape = shape
        self.model = self.generate_model() if weights_path is None else keras.models.load_model(weights_path)

        self.model.compile(
                optimizer = Adam(.0002, .5),
                loss = keras.losses.categorical_crossentropy,
                metrics=['accuracy']
                )

    def predict_key(self, spec):
        return key_arr[np.argmax(self.model.predict(np.expand_dims(self.format_spec(spec), 0)))]

    def format_spec(self, spec):
        if spec.shape[1] > self.shape[1]:
            return spec[:,0:self.shape[1]] # trim
        else:
            temp = np.zeros(shape=self.shape)
            temp[:,0:spec.shape[1]] = spec
            return temp

    def train_on_spec(self, spec, key):
        assert(spec.shape[0] == self.shape[0])
        assert(key in key_arr)

        inp = np.expand_dims(self.format_spec(spec), 0)
        out = np.expand_dims(keras.utils.to_categorical(key_arr.index(key), num_classes=len(key_arr)), 0)

        return self.model.train_on_batch(
                x = inp,
                y = out
                )

    def train_on_spec_arr(self, specs, keys):
        assert(len(keys) == len(specs))

        spec_list = []
        key_list = []

        for spec, key in zip(specs, keys):
            assert(spec.shape[0] == self.shape[0])
            assert(key in key_arr)

            spec_list.append(self.format_spec(spec))
            key_list.append(keras.utils.to_categorical(key_arr.index(key), num_classes=len(key_arr)))

        return self.model.train_on_batch(
                x = np.array(spec_list),
                y = np.array(key_list)
                )

    def generate_model(self):
        inp = Input(shape=self.shape)

        # expand
        for i, filters in enumerate(filter_num):
            model = Conv2D(filters, kernel_size=(5, 5), strides=(1, 1), padding='same')(inp if i == 0 else model)
            model = BatchNormalization(momentum=bn_momentum)(model)
            model = PReLU()(model)
            model = Dropout(rate=dropout_rate)(model)

        # condense
        for i, filters in enumerate(filter_num[::-1]):
            model = Conv2D(filters, kernel_size=(5, 5), strides=(1, 1), padding='same')(model)
            model = BatchNormalization(momentum=bn_momentum)(model)
            model = PReLU()(model)
            model = Dropout(rate=dropout_rate)(model)

        model = MaxPooling2D()(model)
        model = Dropout(rate=dropout_rate)(model)

        model = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(model)
        model = BatchNormalization(momentum=bn_momentum)(model)
        model = PReLU()(model)
        model = Dropout(rate=dropout_rate)(model)

        model = Flatten()(model)

        for nodes in [80, 50]:
            model = Dense(nodes, kernel_initializer='random_normal')(model)
            model = BatchNormalization(momentum=bn_momentum)(model)
            model = PReLU()(model)
            model = Dropout(rate=dropout_rate)(model)

        model = Dense(len(key_arr), kernel_initializer='random_normal', activation='softmax')(model)

        return Model(inp, model)
