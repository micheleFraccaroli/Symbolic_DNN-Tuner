import re
from time import time

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers as reg
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input,
                                     BatchNormalization)
from tensorflow.keras.optimizers import *
from tensorflow_core.python.keras.optimizer_v2.rmsprop import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datasets.cifar_dataset import cifar_data
from LOLR import Lolr
from search_space import search_space


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
        self.counter_fc = 0
        self.counter_conv = 0
        self.tot_fc = 3
        self.tot_conv = 1
        self.rgl = False
        self.dense = False
        self.conv = False

    def build_network(self, params, new):
        """
        Function for define the network structure
        :return: model
        """
        print(self.train_data.shape)

        inputs = Input((self.train_data.shape[1:]))
        x = Conv2D(params['unit_c1'], (3, 3), padding='same')(inputs)
        x = Activation(params['activation'])(x)
        x = Conv2D(params['unit_c1'], (3, 3))(x)
        x = Activation(params['activation'])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        #x = Dropout(params['dr1_2'])(x)

        x = Conv2D(params['unit_c2'], (3, 3), padding='same')(x)
        x = Activation(params['activation'])(x)
        x = Conv2D(params['unit_c2'], (3, 3))(x)
        x = Activation(params['activation'])(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(params['dr1_2'])(x)

        x = Flatten()(x)
        x = Dense(params['unit_d'])(x)
        x = Activation(params['activation'])(x)
        x = Dropout(params['dr_f'])(x)
        x = Dense(self.n_classes)(x)
        x = Activation('softmax')(x)

        model = Model(inputs=inputs, outputs=x)

        model_json = model.to_json()
        model_name = "Model/model-{}.json".format(time())
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)

        return model

    def insert_layer(self, model, layer_regex, params, num_fc=0, num_cv=0, position='after'):
        check = True
        # Auxiliary dictionary to describe the network graph
        K.clear_session()
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
        # Set the input layers of each layer
        for layer in model.layers:
            if self.rgl:
                if 'conv' in layer.name:
                    layer.kernel_regularizer = reg.l2(params['reg'])
            for node in layer.outbound_nodes:
                if len(layer.outbound_nodes) > 1:
                    if 'dense' in layer.name and not self.dense and check:
                        check = False
                        pass
                    else:
                        layer_name = node.outbound_layer.name
                        if layer_name not in network_dict['input_layers_of']:
                            network_dict['input_layers_of'].update(
                                {layer_name: [layer.name]})
                        else:
                            network_dict['input_layers_of'][layer_name].append(layer.name)
                        check = True
                        break
                else:
                    layer_name = node.outbound_layer.name
                    if layer_name not in network_dict['input_layers_of']:
                        network_dict['input_layers_of'].update(
                            {layer_name: [layer.name]})
                    else:
                        network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

        d = network_dict['input_layers_of']
        k = []
        _d = {}
        for i in d.keys():
            k.append(i)
        __d = d.copy()
        for i, j in zip(d, k):
            if len(d[i]) > 1:
                k.insert(k.index(i), d[i][1])
                __d[d[i][1]] = [d[i][0]]
                __d[i].pop(0)
        for i in k:
            _d[i] = __d[i]
        network_dict.pop('input_layers_of')
        network_dict['input_layers_of'] = _d

        # Iterate over all layers after the input
        for layer in model.layers[1:]:
            if layer.name == 'final':
                name = "dense_1"
            else:
                name = layer.name
            layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                           for layer_aux in network_dict['input_layers_of'][name]]
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
                    if self.rgl:
                        x = BatchNormalization()(x)
                    elif self.dense and self.counter_fc < self.tot_fc:
                        if num_fc > 0:
                            if num_fc > self.tot_fc:
                                num_fc = 0
                            for _ in range(num_fc):
                                self.counter_fc += 1
                                x = Dense(params['new_fc'], name='dense_{}'.format(time()))(x)
                        else:
                            x = Dense(params['new_fc'], name='dense_{}'.format(time()))(x)
                        self.last_dense = x.shape[1]
                    elif self.conv and self.counter_conv < self.tot_conv:
                        self.counter_conv += 1
                        naming = '{}'.format(time())
                        x = Conv2D(params['unit_c2'], (3, 3), padding='same', name='conv_{}'.format(naming))(x)
                        x = Activation(params['activation'], name='activation_{}'.format(naming))(x)
                        x = Conv2D(params['unit_c2'], (3, 3), name='conv_2_{}'.format(naming))(x)
                        x = Activation(params['activation'], name='activation_2_{}'.format(naming))(x)
                        x = MaxPooling2D(pool_size=(2, 2), name='maxpooling_{}'.format(naming))(x)
                        x = Dropout(params['dr1_2'], name='dropout_{}'.format(naming))(x)

                if position == 'before':
                    x = layer(x)
            else:
                if layer.output.shape[1] == 10 and re.match(layer_regex, layer.name) and not 'Softmax' in \
                                                                                             layer.output.op.inputs._inputs[
                                                                                                 0].name:
                    x = Dense(layer.output.shape[1], name='final')(x)
                else:
                    if self.conv and 'dense' in layer.name and layer.output.shape[1] == 10:
                        x = Dense(layer.output.shape[1], name='final')(x)
                    elif self.conv and 'dense' in layer.name and not '_' in layer.name:
                        x = Dense(params['unit_d'])(x)
                    else:
                        x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})
        input = model.inputs
        return Model(inputs=input, outputs=x)

    # def lolr_checking(self, mdl, space, lr, batch, trd, trl, ted, tel):
    #     for hp in space:
    #         if hp.name == 'learning_rate':
    #             min_lr = 10 ** -3
    #             max_lr = hp.high
    #
    #     lolr = Lolr(min_lr, max_lr, steps_per_epoch=np.ceil(len(trd) / batch))
    #
    #     adam = Adamax(lr=min_lr)
    #     mdl.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #     mdl.fit(trd, trl, epochs=1, batch_size=batch, verbose=1, validation_data=(ted, tel),
    #             callbacks=[lolr])
    #     return lolr.losses, lolr.lrs

    def training(self, params, new, new_fc, new_conv, da, space):
        """
        Function for compiling and running training
        :return: training history
        """

        model = self.build_network(params, new)
        if new or new_fc or new_conv:
            if new_fc:
                if new_fc[0]:
                    self.dense = True
                    model = self.insert_layer(model, '.*dense.*', params, num_fc=new_fc[1])
            if new:
                self.rgl = True
                self.dense = False
                model = self.insert_layer(model, '.*activation.*', params)
            if new_conv:
                if new_conv[0]:
                    self.conv = True
                    self.dense = False
                    self.rgl = False
                    model = self.insert_layer(model, '.*flatten.*', params, num_cv=new_conv[1],
                                              position='before')

        print(model.summary())
        try:
            model.load_weights("Weights/weights.h5")
        except:
            print("Restart\n")

        # tensorboard logs
        tensorboard = TensorBoard(log_dir="log_folder/logs/{}-{}".format(time(), params['learning_rate']))

        # losses, lrs = self.lolr_checking(model, space, params['learning_rate'], params['batch_size'], self.train_data,
        #                                  self.train_labels, self.test_data, self.test_labels)

        # compiling and training
        _opt = params['optimizer'] + "(learning_rate=" + str(params['learning_rate']) + ")"
        opt = eval(_opt)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        es1 = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=15, verbose=1, mode='min')
        es2 = EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=15, verbose=1, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-4)

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
                verbose=1, validation_data=(self.test_data, self.test_labels),
                callbacks=[tensorboard, reduce_lr, es1, es2]).history
        else:

            history = model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=params['batch_size'],
                                verbose=1,
                                validation_data=(self.test_data, self.test_labels),
                                callbacks=[tensorboard, reduce_lr, es1, es2]).history

        score = model.evaluate(self.test_data, self.test_labels)
        weights_name = "Weights/weights-{}.h5".format(time())
        model.save_weights(weights_name)
        return score, history, model  # , rta


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
    ss = search_space()
    space = ss.search_sp()

    # default_params = {'unit_c1': 57, 'dr1_2': 0.1256695669329692, 'unit_c2': 124, 'unit_d': 315,
    #                   'dr_f': 0.045717734023783346, 'learning_rate': 0.08359864897019328, 'batch_size': 252,
    #                   'optimizer': 'RMSProp', 'activation': 'elu', 'reg': 0.05497168445820486, 'new_fc': 497}

    default_params = {'unit_c1': 32, 'dr1_2': 0.1256695669329692, 'unit_c2': 64, 'unit_d': 315,
                      'dr_f': 0.045717734023783346, 'learning_rate': 0.08359864897019328, 'batch_size': 252,
                      'optimizer': 'RMSProp', 'activation': 'elu', 'reg': 0.05497168445820486, 'new_fc': 497}

    n = neural_network(X_train, Y_train, X_test, Y_test, n_classes)

    score, history, model = n.training(default_params, True, [True, 1], None, None, space)

    f2 = open("algorithm_logs/history.txt", "w")
    f2.write(str(history))
    f2.close()
