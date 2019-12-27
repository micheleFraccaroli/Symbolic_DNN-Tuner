import numpy as np
from neural_network import neural_network as cnet
from dataset.cifar_dataset import cifar_data
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from tensorflow.keras import backend as K
from colors import colors
from controller import controller

'''
hp → hyperparameter space
fmin → minimizing the function
tpe → Tree-structured Parzen Estimator model (optimization algorithm)
'''

class main_class:
    def __init__(self, search_space, X_train, Y_train, X_test, Y_test, n_classes):
        self.space = search_space
        self.algoritm = tpe.suggest()
        self.traials = Trials()
        self.log = open("hyper-parameters.txt", "a")
        self.controller = controller(search_space, self.log, X_train, Y_train, X_test, Y_test, n_classes)
        self.max_evals = 10

    def start(self, search_space):
        '''
        Starting training for Bayesian Optimization
        :return: dict for hyperopt - bayesian optimization
        '''
        self.space, model, to_optimize = self.controller.training(search_space)
        return {'loss': to_optimize, 'status': STATUS_OK}

    def bayesian(self):
        # optimization
        print(colors.WARNING, "\n------------ START BAYESIAN OPTIMIZATION ------------\n", colors.ENDC)
        print(colors.WARNING, "OPTIMIZING ...", colors.ENDC)
        best = fmin(self.start, self.space, algo=self.algoritm, max_evals=self.max_evals)
        return best


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, n_classes = cifar_data()
    # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, stratify=Y_train)
    # default_params = {
    #     'unit_c1': 32,
    #     'dr1_2': 0.25,
    #     'unit_c2': 64,
    #     'unit_d': 512,
    #     'dr_f': 0.5,
    #     'learning_rate': 0.002
    # }

    # hyper-parameters
    search_space = {
        'unit_c1': hp.choice('unit_c1', np.arange(16, 65, 16)),
        'dr1_2': hp.uniform('dr1_2', 0.002, 0.3),
        'unit_c2': hp.choice('unit_c2', np.arange(64, 129, 16)),
        'unit_d': hp.choice('unit_d', np.arange(256, 1025, 256)),
        'dr_f': hp.uniform('dr_f', 0.3, 0.5),
        'learning_rate': hp.loguniform('learning_rate', -10, 0),
    }

    launch = main_class(search_space, X_train, Y_train, X_test, Y_test, n_classes)
