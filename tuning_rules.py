import re
import random as ra
from tensorflow.keras import Model
from tensorflow.keras import regularizers as reg
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K


class tuning_rules:
    def __init__(self, params, ss):
        self.space = params
        self.ss = ss
        # self.weight_decay = 1e-4
        self.count_lr = 0
        self.count_br = 0

    # def insert_layer(self, model, layer_regex, position='after'):
    #     # Auxiliary dictionary to describe the network graph
    #     network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    #     # Set the input layers of each layer
    #     for layer in model.layers:
    #         if 'conv' in layer.name:
    #             layer.kernel_regularizer = reg.l2(self.weight_decay)
    #         for node in layer.outbound_nodes:
    #             layer_name = node.outbound_layer.name
    #             if layer_name not in network_dict['input_layers_of']:
    #                 network_dict['input_layers_of'].update(
    #                     {layer_name: [layer.name]})
    #             else:
    #                 network_dict['input_layers_of'][layer_name].append(layer.name)
    #
    #     # Set the output tensor of the input layer
    #     network_dict['new_output_tensor_of'].update(
    #         {model.layers[0].name: model.input})
    #
    #     # Iterate over all layers after the input
    #     for layer in model.layers[1:]:
    #         layer_input = [network_dict['new_output_tensor_of'][layer_aux]
    #                        for layer_aux in network_dict['input_layers_of'][layer.name]]
    #         if len(layer_input) == 1:
    #             layer_input = layer_input[0]
    #
    #         # Insert layer if name matches the regular expression
    #         if re.match(layer_regex, layer.name):
    #             if position == 'replace':
    #                 x = layer_input
    #             elif position == 'after':
    #                 x = layer(layer_input)
    #             elif position == 'before':
    #                 pass
    #             else:
    #                 raise ValueError('position must be: before, after or replace')
    #
    #             x = BatchNormalization()(x)
    #
    #             if position == 'before':
    #                 x = layer(x)
    #         else:
    #             x = layer(layer_input)
    #
    #         # Set new output tensor (the original one, or the one of the inserted layer)
    #         network_dict['new_output_tensor_of'].update({layer.name: x})
    #     input = model.inputs
    #     del model
    #     K.clear_session()
    #     return Model(inputs=input, outputs=x)

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
                #model = self.insert_layer(model, '.*activation.*')
                model = 'batch'
                print("I've try to fix OVERFITTING by adding regularization and batch normalization\n")
                # model_json = model.to_json()
                # model_name = "Model/model.json"
                # with open(model_name, 'w') as json_file:
                #     json_file.write(model_json)
                controller.set_case(True)
        if "underfitting" in diseases:
            prob = ra.random()
            if prob <= 0.3:
                self.count_lr += 1
                if self.count_lr < 3:
                    for hp in self.space:
                        if hp.name == 'learning_rate':
                            hp.high = params['learning_rate'] + (params['learning_rate']/2)
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
