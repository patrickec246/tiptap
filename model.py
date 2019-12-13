from keras.models import Model
from keras.layers import Activation, Conv2D, Input, Dense, Flatten, BatchNormalization, PReLU, LeakyReLU, Dropout, MaxPooling2D, Reshape
from keras.optimizers import Adam
from keras.activations import softmax

import os
import string
import keras

import skimage.io as io
import numpy as np

# model settings
dropout_rate = 0.2
bn_momentum = 0.8
filter_num = [16, 32, 64, 128]

# detectable keys
detectable_keys = [None]
detectable_keys.extend(list(string.ascii_lowercase))
detectable_keys.extend(list(string.digits))

temp_shape = (513, 173, 1)
inputs = np.ones(shape=temp_shape)

# classifies an audio shape into a key
class TapModel(object):
    def __init__(self, input_shape=temp_shape, batch_num=1, weights_path=None):
        self.output_shape = (batch_num, len(detectable_keys))
        self.input_shape = input_shape

        if weights_path is not None and os.path.exists(weights_path):
            self.model = keras.models.load_model(weights_path)
        else:
            self.model = self.generate_model()

        self.model.compile(
                optimizer = Adam(.0002, .5),
                loss = keras.losses.categorical_crossentropy,
                metrics=['accuracy']
                )

    def predict_keys(self, spec):
        prediction = self.model.predict(np.expand_dims(self.format_spec(spec), 0))[0]

        keys = []
        for column in range(prediction.shape[0]):
            keys.append(detectable_keys[np.argmax(prediction[column])])
        return keys

    def format_spec(self, spec):
        if spec.shape[1] > self.input_shape[1]:
            return spec[:,0:self.input_shape[1]] # trim
        else:
            temp = np.zeros(shape=self.input_shape)
            temp[:,0:spec.shape[1]] = spec
            return temp

    def train_on_spec(self, spec, key):
        assert(spec.shape[0] == self.input_shape[0])

        inp = np.expand_dims(np.expand_dims(spec, 2), 0)
        out = np.expand_dims(key, 0)

        return self.model.train_on_batch(x = inp,y = out)

    def train_on_spec_arr(self, specs, keys):
        assert(len(keys) == len(specs))

        spec_list = []
        key_list = []

        for spec, key in zip(specs, keys):
            assert(spec.shape[0] == self.input_shape[0])
            assert(key in key_arr)

            spec_list.append(self.format_spec(spec))
            key_list.append(keras.utils.to_categorical(key_arr.index(key), num_classes=len(key_arr)))

        return self.model.train_on_batch(
                x = np.array(spec_list),
                y = np.array(key_list)
                )

    def generate_model(self):
        def axis_soft_max(x):
            return softmax(x, axis=0)

        inp = Input(shape=self.input_shape)

        for i, filters in enumerate(filter_num):
            model = Conv2D(filters, kernel_size=(5, 5), strides=(1, 1))(inp if i == 0 else model)
            model = BatchNormalization(momentum=bn_momentum)(model)
            model = LeakyReLU(alpha=0.2)(model)
            model = Dropout(rate=dropout_rate)(model)

        for i in range(3):
            model = MaxPooling2D()(model)
            model = BatchNormalization(momentum=bn_momentum)(model)
            model = LeakyReLU(alpha=0.2)(model)
            model = Dropout(rate=dropout_rate)(model)
        
        model = MaxPooling2D()(model)
        model = Dropout(rate=dropout_rate)(model)

        model = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(model)
        model = BatchNormalization(momentum=bn_momentum)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(rate=dropout_rate)(model)

        model = Flatten()(model)
        model = Dense(512, activation='sigmoid')(model)
        
        model = Dense(self.output_shape[0] * len(detectable_keys), kernel_initializer='random_normal')(model)
        model = Reshape(self.output_shape)(model)
        model = Activation('softmax')(model)

        return Model(inp, model)