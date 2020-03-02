import re
import numpy as np
import sys
import matplotlib.pyplot as plt
from time import time

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers as reg
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input,
                                     BatchNormalization)
from tensorflow.keras.optimizers import *
from tensorflow_core.python.keras.optimizer_v2.rmsprop import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datasets.cifar_dataset import cifar_data
from LOLR import Lolr


class neural_network:
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes):
        self.train_data = X_train
        self.train_labels = Y_train
        self.test_data = X_test
        self.test_labels = Y_test
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        self.train_data /= 255
        self.test_data /= 255
        self.n_classes = n_classes
        self.epochs = 200
        self.last_dense = 0

    def build_network(self, params, new):
        """
        Function for define the network structure
        :return: model
        """
        print(self.train_data.shape)

        inputs = Input((self.train_data.shape[1:]))
        x = Conv2D(params['unit_c1'], (3, 3), padding='same')(inputs)
        x = Activation('relu')(x)
        x = Conv2D(params['unit_c1'], (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(params['unit_c2'], (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(params['unit_c2'], (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(params['dr1_2'])(x)

        x = Flatten()(x)
        x = Dense(params['unit_d'])(x)
        x = Activation('relu')(x)
        x = Dropout(params['dr_f'])(x)
        x = Dense(self.n_classes)(x)
        x = Activation('softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        model_json = model.to_json()
        model_name = "Model/model-{}.json".format(time())
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)

        return model

    def insert_layer(self, model, layer_regex, params, dense, num=0, position='after'):
        # Auxiliary dictionary to describe the network graph
        K.clear_session()
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
        # Set the input layers of each layer
        for layer in model.layers:
            if not dense:
                if 'conv' in layer.name:
                    layer.kernel_regularizer = reg.l2(params['reg'])
            for node in layer.outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

        # Iterate over all layers after the input
        for layer in model.layers[1:]:
            layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                           for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            # Insert layer if name matches the regular expression
            if re.match(layer_regex, layer.name) and layer.output.shape[1] != 10:
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')
                if not 'Softmax' in layer.output.op.inputs._inputs[0].name:
                    if not dense:
                        x = BatchNormalization()(x)
                    else:
                        if num > 0:
                            for n in range(num):
                                x = Dense(params['new_fc'], name='dense_{}'.format(time()))(x)
                        else:
                            x = Dense(params['new_fc'], name='dense_{}'.format(time()))(x)
                        self.last_dense = x.shape[1]

                if position == 'before':
                    x = layer(x)
            else:
                if layer.output.shape[1] == 10 and re.match(layer_regex, layer.name):
                    x = Dense(layer.output.shape[1], name='final')(x)
                else:
                    x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})
        input = model.inputs
        return Model(inputs=input, outputs=x)

    def lolr_checking(self, mdl, space, lr, batch, trd, trl, ted, tel):
        for hp in space:
            if hp.name == 'learning_rate':
                min_lr = hp.low
                max_lr = hp.high

        lolr = Lolr(min_lr, max_lr, steps_per_epoch=np.ceil(len(trd) / batch))

        adam = Adamax(lr=min_lr)
        mdl.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        mdl.fit(trd, trl, epochs=1, batch_size=batch, verbose=1, validation_data=(ted, tel),
                callbacks=[lolr])
        return lolr.losses, lolr.lrs

    def training(self, params, new, new_fc, da, space):
        """
        Function for compiling and running training
        :return: training history
        """
        print("\n-----------------------------------------------------------\n")
        print(params)
        print("\n-----------------------------------------------------------\n")
        model = None

        model = self.build_network(params, new)
        if new or new_fc:
            if new_fc is not None:
                if new_fc[0]:
                    model = self.insert_layer(model, '.*dense.*', params, True, num=new_fc[1])
            if new:
                model = self.insert_layer(model, '.*activation.*', params, False)
        print(model.summary())
        try:
            model.load_weights("Weights/weights.h5")
        except:
            print("Restart\n")

        # tensorboard logs
        tensorboard = TensorBoard(log_dir="log_folder/logs/{}-{}".format(time(), params['learning_rate']))

        # losses, lrs = self.lolr_checking(model, space, params['learning_rate'], params['batch_size'], self.train_data,
        #                                  self.train_labels, self.test_data, self.test_labels)
        # plt.xscale('log')
        # plt.xlabel('learning_rate')
        # plt.ylabel('loss')
        # plt.plot(lrs, losses)

        # compiling and training
        #adam = Adam(lr=params['learning_rate'])
        _opt = params['optimizer'] + "(learning_rate=" + str(params['learning_rate']) + ")"
        opt = eval(_opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')

        if da:
            datagen = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=True)
            datagen.fit(self.train_data)

            history = model.fit_generator(
                datagen.flow(self.train_data, self.train_labels, batch_size=params['batch_size']), epochs=self.epochs,
                verbose=1, validation_data=(self.test_data, self.test_labels), callbacks=[tensorboard, es]).history
        else:

            history = model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=params['batch_size'],
                                verbose=1,
                                validation_data=(self.test_data, self.test_labels),
                                callbacks=[tensorboard, es]).history

        score = model.evaluate(self.test_data, self.test_labels)
        weights_name = "Weights/weights-{}.h5".format(time())
        model.save_weights(weights_name)
        return score, history, model  # , rta


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, n_classes = cifar_data()

    default_params = {'unit_c1': 37, 'dr1_2': 0.08075225090559862, 'unit_c2': 97, 'unit_d': 436,
                      'dr_f': 0.18413154855938407, 'learning_rate': 0.03504090438475931, 'batch_size': 256,
                      'reg': 0.028173467805020478, 'new_fc': 257}

    n = neural_network(X_train, Y_train, X_test, Y_test, n_classes)

    score, history, model = n.training(default_params, False, [True, 8], None)
    print(model.summary())
    f2 = open("algorithm_logs/history.txt", "w")
    f2.write(str(history))
    f2.close()
