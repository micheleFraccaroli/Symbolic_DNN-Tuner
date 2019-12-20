from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *

class tuning_rules:
    def __init__(self, diseases, search_space):
        self.diseases = diseases
        self.space = search_space

    def repair(self):
        if "overfitting" in self.diseases:
            pass
            # do dataaugmentation
            # do regularization
        elif "underfitting" in self.diseases:
            pass
            # add new layers or neurons
            # extend training epochs

        return self.space
