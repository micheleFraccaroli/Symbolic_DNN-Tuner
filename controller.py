from __future__ import print_function

from neural_network import neural_network
from diagnosis import diagnosis
from tuning_rules import tuning_rules
from colors import colors
from search_space import search_space
from tensorflow.keras import backend as K


class controller:
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes):
        # self.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.n_classes = n_classes
        self.d = diagnosis()
        self.ss = search_space()
        self.space = self.ss.search_sp()
        self.tr = tuning_rules(self.space, self.ss)
        self.issues = []
        self.new = None
        self.model = None
        self.params = None

    def set_case(self, new):
        self.new = new

    def training(self, params):
        '''
        Training and tasting the neural network
        training(Train, Labels_train, Test, Label_test)
        :return: model and training historyself.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
        '''
        self.params = params
        print(colors.OKBLUE, "|  --> START TRAINING\n", colors.ENDC)
        K.clear_session()
        self.nn = neural_network(self.X_train, self.Y_train, self.X_test, self.Y_test, self.n_classes)
        self.score, self.history, self.model = self.nn.training(params, self.new, self.model)

        return -self.score[1]

    def diagnosis(self):
        '''
        method for diagnose possible issue like overfitting
        :return: call to tuning method or hp_space, model and accuracy(*-1)
        '''
        print(colors.CYAN, "| START DIAGNOSIS ----------------------------------  |\n", colors.ENDC)
        diagnosis_logs = open("algorithm_logs/diagnosis_logs.txt", "a")
        self.d.reset_diagnosis()
        self.issues = self.d.diagnosis(self.history, self.score, diagnosis_logs)
        diagnosis_logs.close()
        print(colors.CYAN, "| END DIAGNOSIS   ----------------------------------  |\n", colors.ENDC)

        if self.issues:
            self.space, to_optimize, self.model = self.tuning()
            return self.space, to_optimize
        else:
            return self.space, -self.score[1]

    def tuning(self):
        '''
        tuning the hyper-parameter space or add new hyper-parameters
        :return: new hp_space, new_model and accuracy(*-1) for the Bayesian Optimization
        '''
        print(colors.FAIL, "| START TUNING    ----------------------------------  |\n", colors.ENDC)
        tuning_logs = open("algorithm_logs/tuning_logs.txt", "a")
        new_space, self.model = self.tr.repair(self, self.issues, tuning_logs, self.model, self.params)
        tuning_logs.close()
        self.issues = []
        print(colors.FAIL, "| END TUNING      ----------------------------------  |\n", colors.ENDC)

        return new_space, -self.score[1], self.model
