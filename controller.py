from neural_network import neural_network
from tuning_rules import *


class controller:
    def __init__(self, search_space, output_file):
        self.space = search_space
        self.f = output_file
        self.logs = open("issues_log.txt","a")

    def training(self, X_train, Y_train, X_test, Y_test):
        '''
        Training and tasting the neural network
        training(Train, Labels_train, Test, Label_test)
        return model and training history
        '''
        nn = neural_network(X_train, Y_train, X_test, Y_test, self.space, self.f)

        self.model, self.history = nn.training()
        self.score = self.model.evaluate(X_test, Y_test)

        return self.model, -self.score[1]


    def tuning(self):
        '''
        tuning che hyperparameter space or add new hyperparameters
        '''

        tr = tuning_rules(self.issues, self.space)

        # write the issues founded on log file
        self.logs.write(str(self.issues) + "\n\n")
        self.logs.close()
        self.issues = []

        new_space = tr.repair()

        return new_space