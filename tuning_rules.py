from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers as reg
from tensorflow.keras.layers import *
from search_space import search_space

class tuning_rules:
    def __init__(self, diseases, search_space, tuning_logs, model):
        self.diseases = diseases
        self.space = search_space
        self.tuning_logs = tuning_logs
        self.weight_decay = 1e-4
        self.model = model

    def repair(self):
        '''
        Method for fix the issues
        :return: new hp_space and new model
        '''
        if "overfitting" in self.diseases:
            self.tuning_logs.write("Applied regulatization and batch normalization after activation\n")
            for l in self.model.layers:
                if 'conv' in l.name:
                    l.kernel_regularize = reg.l2(self.weight_decay)
                if 'activation' in l.name:
                    self.model.insert(self.model.layers.index(l) + 1, BatchNormalization())
        elif "underfitting" in self.diseases:
            pass
            # add new layers or neurons
            # extend training epochs

        return self.space, self.model
