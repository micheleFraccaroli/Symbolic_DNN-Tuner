import sys
from tensorflow.keras import backend as K
from neural_network import neural_network
from diagnosis import diagnosis
from tuning_rules import tuning_rules
from colors import colors
from search_space import search_space
import datetime


class controller:
    def __init__(self, output_file, X_train, Y_train, X_test, Y_test, n_classes, first_params):
        self.f = output_file
        self.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes, self.f)
        self.model = self.nn.build_network(first_params)
        self.X_test = X_test[:100]
        self.Y_test = Y_test[:100]
        self.issues = []
        self.ss = search_space()
        self.space = self.ss.search_sp()

    def training(self, params):
        '''
        Training and tasting the neural network
        training(Train, Labels_train, Test, Label_test)
        :return: model and training history
        '''

        print(colors.OKBLUE, "|  --> START TRAINING\n", colors.ENDC)
        self.model, self.history = self.nn.training(params, self.model)
        self.f.close()
        self.score = self.model.evaluate(self.X_test, self.Y_test)

        return -self.score[1]

    def diagnosis(self):
        '''
        method for diagnose possible issue like overfitting
        :return: call to tuning method or hp_space, model and accuracy(*-1)
        '''
        print(colors.CYAN, "|  ----> START DIAGNOSIS\n", colors.ENDC)
        diagnosis_logs = open("algorithm_logs/diagnosis_logs.txt", "a")
        dt = datetime.datetime.now()
        diagnosis_logs.write("-------------- " + str(dt) + " --------------\n")
        d = diagnosis(self.history, diagnosis_logs, self.score)
        self.issues = d.diagnosis()
        diagnosis_logs.close()

        if self.issues:
            self.space, self.model, to_optimize = self.tuning()
            return self.space, self.model, to_optimize
        else:
            return self.space, self.model, -self.score[1]

    def tuning(self):
        '''
        tuning the hyper-parameter space or add new hyper-parameters
        :return: new hp_space, new_model and accuracy(*-1) for the Bayesian Optimization
        '''
        print(colors.FAIL, "|  ------> START TUNING\n", colors.ENDC)
        tuning_logs = open("algorithm_logs/tuning_logs.txt", "a")
        dt = datetime.datetime.now()
        tuning_logs.write("-------------- " + str(dt) + " --------------\n")
        tr = tuning_rules(self.issues, self.space, self.ss, tuning_logs, self.model)
        new_space, new_model = tr.repair()
        tuning_logs.close()
        self.issues = []
        #K.clear_session()
        return new_space, new_model, -self.score[1]
