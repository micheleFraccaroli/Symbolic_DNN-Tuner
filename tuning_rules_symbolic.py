import random as ra
import sys

class tuning_rules_symbolic:
    def __init__(self, params, ss):
        self.space = params
        self.ss = ss
        self.count_lr = 0
        self.count_br = 0

    def reg_l2(self, controller):
        self.count_br += 1
        if self.count_br <= 1:
            print("I've try to fix OVERFITTING by adding regularization and batch normalization\n")
            model = 'batch'
            controller.set_case(True)
            new_p = {'reg': 1e-4}
            self.space = self.ss.add_params(new_p)

    def incr_dropout(self, params):
        for hp in self.space:
            if 'dr' in hp.name:
                hp.low = params[hp.name] - 1

    def decr_lr(self, params):
        for hp in self.space:
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + (params['learning_rate'] / 2)

    def inc_neurons(self, params):
        for hp in self.space:
            if 'unit_c1' in hp.name:
                hp.low = params['unit_c1'] - 1
            if 'unit_c2' in hp.name:
                hp.low = params['unit_c2'] - 1
            if 'unit_d' in hp.name:
                hp.low = params['unit_d'] - 1

    def inc_batch_size(self, params):
        for hp in self.space:
            if hp.name == 'batch_size':
                hp.low = params['batch_size'] - 1

    def repair(self, controller, sym_diseases, model, params):
        '''
        Method for fix the issues
        :return: new hp_space and new model
        '''
        del controller.model
        for d in sym_diseases:
            if d != 'reg_l2':
                d = "self." + d + "(params)"
            print(d)
            eval(d)

        return self.space, model
