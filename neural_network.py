import re
from time import time
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input,
                                     BatchNormalization)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import model_from_json
from dataset.cifar_dataset import cifar_data

class neural_network:
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes):
        self.train_data = X_train[:10000]
        self.train_labels = Y_train[:10000]
        self.test_data = X_test[:6000]
        self.test_labels = Y_test[:6000]
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        self.train_data /= 255
        self.test_data /= 255
        self.n_classes = n_classes
        self.epochs = 30
        self.batch_size = 32

    def build_network(self, params, new):
        '''
        Function for define the network structure
        :return: model
        '''

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

    def build_new(self, old_model, new_model):
        inputs = Input((self.train_data.shape[1:]))
        x = inputs
        for i, j in zip(old_model.layers, new_model.layers):
            old = " ".join(re.findall("[a-zA-Z]+", i.name))
            new = " ".join(re.findall("[a-zA-Z]+", j.name))
            if 'conv' in new:
                regularizer = j.kernel_regularizer
            if old != 'input':
                if 'conv' in old:
                    try:
                        i.kernel_regularizer = regularizer
                        x = i(x)
                    except:
                        pass
                elif 'activation' in old:
                    if 'Softmax' in i.output.name:
                        x = i(x)
                    else:
                        x = i(x)
                        x = BatchNormalization()(x)
                else:
                    x = i(x)
                # else:
                #     x = j(x)
                #     x = i(x)
        result_model = Model(inputs=inputs, outputs=x)
        model_json = result_model.to_json()
        model_name = "Model/model.json"
        with open(model_name, 'w') as json_file:
            json_file.write(model_json)

        return result_model

    def training(self, params, new, ):
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
            json_file = open("Model/model.json")
            lmj = json_file.read()
            json_file.close()
            new_model = model_from_json(lmj)
            model = self.build_new(model, new_model)

        print(model.summary())

        # tensorboard logs
        tensorboard = TensorBoard(log_dir="logs_hyperopt/{}-{}".format(params['learning_rate'], time()))

        # compiling and training
        adam = Adam(lr=params['learning_rate'])
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        history = model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size,
                            verbose=1,
                            validation_data=(self.test_data, self.test_labels),
                            callbacks=[tensorboard]).history
        score = model.evaluate(self.test_data, self.test_labels)
        weights_name = "Weights/weights.h5"
        model.save_weights(weights_name)

        return score, history, model


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, n_classes = cifar_data()

    default_params = {'unit_c1': 64, 'dr1_2': 0.27388076426452224, 'unit_c2': 124, 'unit_d': 505,
                      'dr_f': 0.4033067277510234, 'learning_rate': 9.426807887713249e-05}

    n = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
    # model = n.build_network(default_params)
    score, history, model = n.training(default_params, False)
