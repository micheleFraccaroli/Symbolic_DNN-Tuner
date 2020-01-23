import re
import sys
from time import time
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input,
                                     BatchNormalization)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from real_time_analysis import real_time_analysis
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers as reg
from datasets.cifar_dataset import cifar_data

class neural_network:
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes):
        self.train_data = X_train[:1000]
        self.train_labels = Y_train[:1000]
        self.test_data = X_test[:600]
        self.test_labels = Y_test[:600]
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        self.train_data /= 255
        self.test_data /= 255
        self.n_classes = n_classes
        self.epochs = 10
        # self.weight_decay = 1e-4
        # self.batch_size = 96

    def build_network(self, params, new):
        '''
        Function for define the network structure
        :return: model
        '''
        '''
        inputs = Input((self.train_data.shape[1:]))
        x = Conv2D(params['unit_c1'], (3, 3), padding='same', kernel_regularizer=reg.l2(1e-4))(inputs)
        x = Activation('relu')(x)
        x = Conv2D(params['unit_c1'], (3, 3), kernel_regularizer=reg.l2(1e-4))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(params['unit_c2'], (3, 3), padding='same', kernel_regularizer=reg.l2(1e-4))(x)
        x = Activation('relu')(x)
        x = Conv2D(params['unit_c2'], (3, 3), kernel_regularizer=reg.l2(1e-4))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(params['dr1_2'])(x)

        x = Flatten()(x)
        x = Dense(params['unit_d'])(x)
        x = Activation('relu')(x)
        x = Dropout(params['dr_f'])(x)
        x = Dense(self.n_classes)(x)
        x = Activation('softmax')(x)
        '''
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

        if not new:
            model_json = model.to_json()
            model_name = "Model/model.json"
            with open(model_name, 'w') as json_file:
                json_file.write(model_json)

        return model

    def insert_layer(self, model, layer_regex, params, position='after'):
        # Auxiliary dictionary to describe the network graph
        K.clear_session()
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
        # Set the input layers of each layer
        for layer in model.layers:
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
            if re.match(layer_regex, layer.name):
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')
                if not 'Softmax' in layer.output.op.inputs._inputs[0].name:
                    x = BatchNormalization()(x)

                if position == 'before':
                    x = layer(x)
            else:
                x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})
        input = model.inputs
        return Model(inputs=input, outputs=x)

    def training(self, params, new, model_b):
        '''
        Function for compiling and running training
        :return: training history
        '''
        print("\n-----------------------------------------------------------\n")
        print(params)
        print("\n-----------------------------------------------------------\n")
        model = None

        model = self.build_network(params, new)
        if new:
            model = self.insert_layer(model, '.*activation.*', params)
        print(model.summary())
        try:
            model.load_weights("Weights/weights.h5")
        except:
            print("Restart\n")

        # tensorboard logs
        tensorboard = TensorBoard(log_dir="logs/{}-{}".format(time(), params['learning_rate']), update_freq='batch')

        # compiling and training
        adam = Adam(lr=params['learning_rate'])
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        rta = real_time_analysis()
        rta.set_epochs(self.epochs)
        # es = EarlyStopping(monitor='val_loss', mode='min')

        history = model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=params['batch_size'],
                            verbose=1,
                            validation_data=(self.test_data, self.test_labels),
                            callbacks=[tensorboard, rta]).history
        score = model.evaluate(self.test_data, self.test_labels)
        weights_name = "Weights/weights.h5"
        model.save_weights(weights_name)
        return score, history, model, rta


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, n_classes = cifar_data()

    default_params = {'unit_c1': 16, 'dr1_2': 0.002, 'unit_c2': 64, 'unit_d': 512, 'dr_f': 0.5, 'learning_rate': 0.1, 'batch_size': 10}

    n = neural_network(X_train, Y_train, X_test, Y_test, n_classes)

    score, history, model, rta = n.training(default_params, False, None)

    print("\n-----------------------------------------------------------\n")
    print("\n-----------------------------------------------------------\n")
    print(history)
    print("\n-----------------------------------------------------------\n")
    print(rta.accuracy[:100])
    print("\n-----------------------------------------------------------\n")
    print("\n-----------------------------------------------------------\n")

    tot = 0
    for a in rta.accuracy[:100]:
        tot = a + tot

    res = tot / 100
    print(res)
