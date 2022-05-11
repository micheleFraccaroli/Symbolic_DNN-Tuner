import random as ra
import sys
from time import time


class tuning_rules_symbolic:
    def __init__(self, params, ss, controller):
        self.space = params
        self.ss = ss
        self.controller = controller
        self.count_lr = 0
        self.count_da = 0
        self.count_br = 0
        self.count_new_fc = 0
        self.count_new_cv = 0

    def reg_l2(self):
        self.count_br += 1
        if self.count_br <= 1:
            print("I've try to fix OVERFITTING by adding regularization and batch normalization\n")
            model = 'batch'
            self.controller.set_case(True)
            new_p = {'reg': 1e-4}
            self.space = self.ss.add_params(new_p)

    def new_fc_layer(self):
        self.count_new_fc += 1
        self.controller.add_fc_layer(True, self.count_new_fc)
        new_p = {'new_fc': 512}
        self.space = self.ss.add_params(new_p)

    def new_conv_layer(self):
        self.count_new_cv += 1
        self.controller.add_conv_section(True, self.count_new_cv)

    def data_augmentation(self):
        self.controller.set_data_augmentation(True)

    def inc_dropout(self, params):
        self.controller.set_data_augmentation(False)
        for hp in self.space:
            if 'dr' in hp.name:
                hp.low = params[hp.name] - params[hp.name] / 100

    def decr_lr(self, params):
        for hp in self.space:
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + (params['learning_rate'] / 2)

    def inc_lr(self, params):
        for hp in self.space:
            if hp.name == 'learning_rate':
                hp.high = params['learning_rate'] + hp.high

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
                
    # new action for hardware constraints
    def dec_neurons(self, params):
        for hp in self.space:
            if 'unit_c1' in hp.name:
                hp.high = params['unit_c1'] + 1
            if 'unit_c2' in hp.name:
                hp.high = params['unit_c2'] + 1
            if 'unit_d' in hp.name:
                hp.high = params['unit_d'] + 1

    def remove_conv_layer(self):
        self.controller.remove_conv_section(True)
    
    # ------------------------------------

    def repair(self, sym_tuning, model, params):
        '''
        Method for fix the issues
        :return: new hp_space and new model
        '''
        del self.controller.model
        for d in sym_tuning:
            if d != 'reg_l2' and d != 'data_augmentation' and d != 'new_fc_layer' and d != 'new_conv_layer':
                d = "self." + d + "(params)"
            else:
                d = "self." + d + "()"
            eval(d)

        return self.space, model
