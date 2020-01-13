import re
import random as ra
from tensorflow.keras import Model
from tensorflow.keras import regularizers as reg
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class tuning_rules:
    def __init__(self, diseases, params, ss, tuning_logs, model):
        self.diseases = diseases
        self.space = params
        self.ss = ss
        self.tuning_logs = tuning_logs
        self.weight_decay = 1e-4
        self.model = model

    def insert_layer(self, model, layer_regex, position='after'):
        # Auxiliary dictionary to describe the network graph
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

        # Set the input layers of each layer
        for layer in model.layers:
            for node in layer.outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

        # Iterate over all layers after the input
        for layer in model.layers[1:]:
            layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                           for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            # Insert layer if name matches the regular expression
            if re.match(layer_regex, layer.name):
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')

                x = BatchNormalization()(x)

                if position == 'before':
                    x = layer(x)
            else:
                x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})

        return Model(inputs=model.inputs, outputs=x)

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
                    new_hp = {'regularization': self.weight_decay}
                    self.space = self.ss.add_params(new_hp)

            self.model = self.insert_layer(self.model, '.*activation.*')
            print("I've try to fix OVERFITTING\n")
        if "underfitting" in self.diseases:
            prob = ra.random()
            if prob <= 0.5:
                for hp in self.space:
                    if hp.name == 'learning_rate':
                        hp.high = hp.high / 10 ** 1.5
                print("I've try to fix UNDERFITTING\n")
            else:
                for hp in self.space:
                    if 'unit' in hp.name:
                        hp.low = int(hp.low + ((hp.high - hp.low)/2))
            # add new layers or neurons
            # extend training epochs
        return self.space, self.model
