from tensorflow.keras import backend as K
from neural_network import neural_network
from diagnosis import diagnosis
from tuning_rules import tuning_rules
from colors import colors


class controller:
    def __init__(self, output_file, X_train, Y_train, X_test, Y_test, n_classes):
        self.f = output_file
        self.diagnosis_logs = open("diagnosis_logs.txt", "a")
        self.tuning_logs = open("tuning_logs.txt", "a")
        self.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes, self.space, self.f)
        self.X_test = X_test
        self.Y_test = Y_test
        # self.model = self.nn.build_network()

    def training(self, search_space, init=True):
        '''
        Training and tasting the neural network
        training(Train, Labels_train, Test, Label_test)
        :return: model and training history
        '''

        print(colors.MAGENTA, "\n------------ START TRAINING ------------\n", colors.ENDC)
        self.history = self.nn.training(search_space)
        self.f.close()
        self.score = self.model.evaluate(self.X_test, self.Y_test)
        K.clear_session()

        if init:
            return -self.score[1]
        else:
            # -self.score[1] is the opposite of accuracy
            space, model, to_optimize = self.diagnosis()
            return to_optimize

    def diagnosis(self):
        '''
        method for diagnose possible issue like overfitting
        :return: call to tuning method or hp_space, model and accuracy(*-1)
        '''
        print(colors.OKBLUE, "\n------------ START DIAGNOSIS ------------\n", colors.ENDC)
        d = diagnosis(self.history, self.diagnosis_logs)
        self.issues = d.diagnosis()
        self.diagnosis_logs.close()

        if self.issues:
            self.space, self.model, -self.score[1] = self.tuning()
            return self.space, self.model, -self.score[1]
        else:
            return self.space, self.model, -self.score[1]

    def tuning(self):
        '''
        tuning the hyper-parameter space or add new hyper-parameters
        :return: new hp_space, new_model and accuracy(*-1) for the Bayesian Optimization
        '''
        print(colors.OKGREEN, "\n------------ START TUNING ------------\n", colors.ENDC)
        tr = tuning_rules(self.issues, self.space, self.tuning_logs, self.model)
        new_space, new_model = tr.repair()
        self.tuning_logs.close()
        self.issues = []

        return new_space, new_model, -self.score[1]
