from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input, BatchNormalization)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from dataset.cifar_dataset import cifar_data


class neural_network:
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes, f):
        self.train_data = X_train[:500]
        self.train_labels = Y_train[:500]
        self.test_data = X_test[:100]
        self.test_labels = Y_test[:100]
        self.n_classes = n_classes
        self.epochs = 10
        self.batch_size = 32
        self.f = f

    def build_network(self, params):
        '''
        Function for define the network structure
        :return: model
        '''
        self.f.write(str(params) + "\n")

        print("\n-----------------------------------------------------------\n")
        print(params)
        print("\n-----------------------------------------------------------\n")

        inputs = Input((self.train_data.shape[1:]))
        x = Conv2D(params['unit_c1'], (3, 3), padding='same')(inputs)
        x = Activation('relu')(x)
        x = Conv2D(params['unit_c1'], (3,3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

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

        return model

    def training(self, params, model):
        '''
        Function for compiling and running training
        :return: training history
        '''
        #model = self.build_network(params)

        # tensorboard logs
        tensorboard = TensorBoard(log_dir="logs/{}".format(params['learning_rate']))

        # compiling and training
        adam = Adam(lr=params['learning_rate'])
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        self.train_data /= 255
        self.test_data /= 255

        history = model.fit(self.train_data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size,
                            verbose=1,
                            validation_data=(self.test_data, self.test_labels),
                            callbacks=[tensorboard]).history

        return model, history


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test, n_classes = cifar_data()

    default_params = {
        'unit_c1': 64,
        'dr1_2': 0.25,
        'unit_c2': 96,
        'unit_d': 512,
        'dr_f': 0.5,
        'learning_rate': 0.002
    }

    f = open("test/hyperparameters.txt", "w")
    n = neural_network(X_train, Y_train, X_test, Y_test,n_classes, f)
    model, history = n.training(default_params)
    f.close()
    score = model.evaluate(X_test[:100], Y_test[:100])
