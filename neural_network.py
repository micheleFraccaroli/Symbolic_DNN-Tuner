from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from dataset.cifar_dataset import cifar_data


class neural_network:
    def __init__(self, X_train, Y_train, X_test, Y_test, params, f):
        self.train_data = X_train
        self.train_labels = Y_train
        self.test_data = X_test
        self.test_labels = Y_test
        self.params = params
        self.file_out = f
        self.epochs = 50
        self.batch_size = 32

    def build_network(self):
        '''
        Function for define the network structure
        return model
        '''

        print("\n-----------------------------------------------------------\n")
        print(self.params)
        self.f.write(str(self.params) + "\n")
        print("\n-----------------------------------------------------------\n")

        model = Sequential()
        model.add(Conv2D(self.params['unit_c1'], (3, 3), padding='same',
                         input_shape=self.train_data.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(self.params['unit_c1'], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.params['dr1_2']))

        model.add(Conv2D(self.params['unit_c2'], (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.params['unit_c2'], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(self.params['dr1_2']))

        model.add(Flatten())
        model.add(Dense(self.params['unit_d']))
        model.add(Activation('relu'))
        model.add(Dropout(self.params['dr_f']))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        return model

    def training(self):
        '''
        Function for compiling and running training
        return model and training history
        '''

        model = self.build_network()

        # logs
        tensorboard = TensorBoard(log_dir="logs/{}".format(self.params['learning_rate']))

        # Training
        adam = Adam(lr=self.params['learning_rate'])
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
    n = neural_network(X_train, Y_train, X_test, Y_test, default_params, f)
    model, history = n.training()
    f.close()
    score = model.evaluate(X_test, Y_test)

    print(score)
    print("------------------------------------------------------------")
    print(history)
