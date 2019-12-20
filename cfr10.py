from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from cifar_dataset import cifar_data


def net(X_train, Y_train, X_test, Y_test, params, f):
    print("\n-----------------------------------------------------------\n")
    print(params)
    f.write(str(params) + "\n")
    print("\n-----------------------------------------------------------\n")

    model = Sequential()
    model.add(Conv2D(params['unit_c1'], (3, 3), padding='same',
                     input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params['unit_c1'], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dr1_2']))

    model.add(Conv2D(params['unit_c2'], (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(params['unit_c2'], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dr1_2']))

    model.add(Flatten())
    model.add(Dense(params['unit_d']))
    model.add(Activation('relu'))
    model.add(Dropout(params['dr_f']))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # logs
    tensorboard = TensorBoard(log_dir="logs/{}".format(params['learning_rate']))

    # Training
    adam = Adam(lr=params['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, Y_test),
              callbacks=[tensorboard])

    return model


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

    f = open("hyperparameters.txt", "w")
    model = net(X_train, Y_train, X_test, Y_test, default_params, f)
    f.close()
    score = model.evaluate(X_test, Y_test)

    print(score)
