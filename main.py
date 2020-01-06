import numpy as np
from neural_network import neural_network as cnet
from dataset.cifar_dataset import cifar_data
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from tensorflow.keras import backend as K
from scipy.stats import loguniform
from sklearn.gaussian_process import GaussianProcessRegressor
from colors import colors
from controller import controller
from bayesian_opt import bayesian_opt


class main_class:
    def __init__(self, search_space, X_train, Y_train, X_test, Y_test, n_classes):
        self.space = search_space
        # self.algoritm = tpe.suggest()
        # self.traials = Trials()
        self.log = open("hyper-parameters.txt", "a")
        self.controller = controller(self.log, X_train, Y_train, X_test, Y_test, n_classes)
        self.max_evals = 3

    def start(self, space, init=False):
        '''
        Starting training for Bayesian Optimization
        :return: params to optimize
        '''
        to_optimize = self.controller.training(space,init)
        return to_optimize

    def bayOptimization(self):
        '''
        Starting bayesian Optimization
        :return:
        '''
        print(colors.WARNING, "\n------------ START BAYESIAN OPTIMIZATION ------------\n", colors.ENDC)
        print(colors.WARNING, "OPTIMIZING ...", colors.ENDC)
        model = GaussianProcessRegressor()

        bo = bayesian_opt(3)
        init, init1 = bo.choice(self.space)
        init1 = np.array(init1).reshape(-1,1
                                        )
        y = self.start(init, True)
        y = np.array(y)
        y = np.repeat(y,len(init1))
        y = y.reshape(-1,1)

        model.fit(init1,y)


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

    # search_space = {
    #     'unit_c1': hp.choice('unit_c1', np.arange(16, 65, 16)),
    #     'dr1_2': hp.uniform('dr1_2', 0.002, 0.3),
    #     'unit_c2': hp.choice('unit_c2', np.arange(64, 129, 16)),
    #     'unit_d': hp.choice('unit_d', np.arange(256, 1025, 256)),
    #     'dr_f': hp.uniform('dr_f', 0.3, 0.5),
    #     'learning_rate': hp.loguniform('learning_rate', -10, 0),
    # }

    # hyper-parameters
    search_space = {
        'unit_c1': [16,32,48,64],
        'dr1_2': np.random.normal(0.002, 0.3),
        'unit_c2': [64,80,96,112,128],
        'unit_d': [256,512],
        'dr_f': np.random.normal(0.3,0.5),
        'learning_rate': loguniform.rvs(10**-5, 10**-1)
    }
    launch = main_class(search_space, X_train, Y_train, X_test, Y_test, n_classes)


