import random as ra

class tuning_rules:
    def __init__(self, params, ss):
        self.space = params
        self.ss = ss
        # self.weight_decay = 1e-4
        self.count_lr = 0
        self.count_br = 0

    def repair(self, controller, diseases, tuning_logs, model, params):
        '''
        Method for fix the issues
        :return: new hp_space and new model
        '''
        del controller.model
        if "overfitting" in diseases:
            self.count_br += 1
            if self.count_br <= 1:
                tuning_logs.write("Applied regulatization and batch normalization after activation\n")
                print("I've try to fix OVERFITTING by adding regularization and batch normalization\n")
                model = 'batch'
                controller.set_case(True)
                new_p = {'reg': 1e-4}
                self.space = self.ss.add_params(new_p)

        if "underfitting" in diseases:
            prob = ra.random()
            if prob <= 0.3:
                self.count_lr += 1
                if self.count_lr < 3:
                    for hp in self.space:
                        if hp.name == 'learning_rate':
                            hp.high = params['learning_rate'] + (params['learning_rate'] / 2)
                    tuning_logs.write("I've try to fix UNDERFITTING decreasing the learning_rate\n")
            else:
                for hp in self.space:
                    if 'unit_c1' in hp.name:
                        hp.low = params['unit_c1'] - 1
                    if 'unit_c2' in hp.name:
                        hp.low = params['unit_c2'] - 1
                    if 'unit_d' in hp.name:
                        hp.low = params['unit_d'] - 1

                tuning_logs.write("I've try to fix UNDERFITTING increasing the number of the node per layers\n")

        if "increasing_loss" in diseases:
            for hp in self.space:
                if hp.name == 'learning_rate':
                    hp.high = params['learning_rate'] + (params['learning_rate'] / 2)

            tuning_logs.write("I've try to fix INCREASING LOSS decreasing the learning_rate\n")

        if "floating_loss" in diseases:
            for hp in self.space:
                if hp.name == 'batch_size':
                    hp.low = params['batch_size'] - 1
            tuning_logs.write("I've try to fix FLOATING LOSS increasing the batch size\n")

        return self.space, model