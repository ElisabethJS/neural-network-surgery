# Copyright (c) 2021 Schiessler et al.

# Model Adaptations for FFNNs
# Presumption: Model consists only of series of dense layers
# This class provides several options to build a new FFNN consisting only of dense layers
# based on a provided base model
# base assumption: the model will have one input and one output layer, that are not to be modified

import numpy as np
from tensorflow import keras
import copy

class ModelAdapter:
    def __init__(self, base_model, loss, optimizer, metrics, overwrite_base=False, index_shift=0):
        self.parse_model(base_model)
        self.overwrite_base = overwrite_base
        self.compile_optimizer = optimizer
        self.compile_loss = loss
        self.compile_metrics = metrics
        self.index_shift = index_shift

    def parse_model(self, base_model):
        self.base_model = base_model
        self.base_config = base_model.get_config()
        self.all_weights = self.base_model.get_weights()
        self.base_weights = []
        self.base_biases = []

        last_was_weight = False #first will always be a weight term
        for w in self.all_weights:
            if not last_was_weight: #this is a weight term
                self.base_weights.append(w)
                last_was_weight = True
            else:
                if np.ndim(w) > 1:
                    # this is a weight term -> there was no bias term!
                    self.base_biases.append(None)
                    self.base_weights.append(w)
                else:
                    last_was_weight = False
                    self.base_biases.append(w)

        ln = []
        for layer in self.base_model.layers:
            if layer.name.startswith('dense'):
                if len(layer.name) > 5:
                    ln.append(int(layer.name[6:]))  
                else:
                    ln.append(0)
        self.layer_name = max(ln) + 1
        self.layer_count = len(self.base_config['layers'])

    def perform_checks(self, layer_number, neurons_to_remove=0, last_layer_allowed=False):
        result = False
        
        # handle backwards count - todo
        #if layer_number < 0:
        #    layer_number = self.layer_count + layer_number # now -1 equals last_layer
        #    # todo.. need to change layer_number in caller as well - no by ref in python :(
        #    if layer_number < 0:
        #        print("Specified layer number outside of allowed range!")
        #        return result, layer_number
        
        # input layer not allowed
        if layer_number == 0:
            print("Input layer cannot be changed")
            return result, layer_number
        
        # do not count output layer unless specified, 
        if (layer_number == self.layer_count - 1 and last_layer_allowed == False) or (layer_number >= self.layer_count):
            print("Specified layer number %s outside of allowed range!" % layer_number)
            return result, layer_number

        # morde neurons to remove than exist
        if neurons_to_remove > 0 and neurons_to_remove >= self.base_config['layers'][layer_number]['config']['units']:
            print("Cannot remove specified number of neurons!")
            return result, layer_number

        result = True
        return result, layer_number

    def get_config(self):
            return copy.deepcopy(self.base_config)

    def return_compiled_model(self, model):
        if self.overwrite_base:
            self.parse_model(model)
        
        model.compile(loss=self.compile_loss, optimizer=self.compile_optimizer, metrics=self.compile_metrics)
        return model

    def Identity(self, *args, **kwargs):
        '''
        Returns an unlinked copy of the original model
        '''
        new_model = keras.models.clone_model(self.base_model)
        new_model.set_weights(self.all_weights)
        return self.return_compiled_model(new_model)

    
    def AddNeuron(self, layer_number, neurons_to_add=1, overwrite_base=None):
        '''
        Adds a number of units to the specified layer (starting from 1, can't add to input layer)
        '''
        check_ok, layer_number = self.perform_checks(layer_number)
        if not check_ok:
            return None

        if overwrite_base is not None:
            self.overwrite_base = overwrite_base

        new_config = self.get_config()
        
        old_unit_count = new_config['layers'][layer_number]['config']['units']
        new_config['layers'][layer_number]['config']['units'] += neurons_to_add
        use_bias_this = new_config['layers'][layer_number]['config']['use_bias']
        use_bias_next = new_config['layers'][layer_number + 1]['config']['use_bias'] 
        
        new_model = keras.Sequential.from_config(new_config)
        
        # generate adapted weights
        w0 = copy.deepcopy(self.base_weights[layer_number - 1 - self.index_shift]) # input layer has no weights
        w1 = copy.deepcopy(self.base_weights[layer_number - self.index_shift])
        if use_bias_this:
            b0 = copy.deepcopy(self.base_biases[layer_number - 1 - self.index_shift])

        loop_all_times = neurons_to_add // old_unit_count
        loop_remainder = neurons_to_add % old_unit_count
        
        for idx, val in enumerate(loop_remainder*[loop_all_times + 1] + (old_unit_count-loop_remainder)*[loop_all_times]):
            if val > 0:
                w0[:, idx] = w0[:, idx]/(val+1)
                if use_bias_this:
                    b0[idx] = b0[idx]/(val+1)
                for _ in range(val):
                    w0 = np.append(w0, np.transpose([w0[:, idx]]), axis=1)
                    w1 = np.append(w1, [w1[idx, :]], axis=0) # no need for division by value here
                    if use_bias_this:
                        b0 = np.append(b0, [b0[idx]])

        # concat, set weights
        new_weights = []
        for idx in range(layer_number - 1 - self.index_shift):
            new_weights += [self.base_weights[idx]] 
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_weights += [w0] 
        if use_bias_this:
            new_weights += [b0]
        
        new_weights += [w1]
        if use_bias_next:
            new_weights += [self.base_biases[layer_number - self.index_shift]]

        for idx in range(layer_number + 1 - self.index_shift, len(self.base_weights)):
            new_weights += [self.base_weights[idx]]
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_model.set_weights(new_weights)

        return self.return_compiled_model(new_model)

    def AddLayer(self, layer_number, activation=None, use_bias=True, overwrite_base=None):
        '''
        Adds a layer at the specified index. Output layer will always remain at last index
        Number of units will be equal to number of units in the following layer
        '''
        check_ok, layer_number = self.perform_checks(layer_number)
        if not check_ok:
            return None

        if overwrite_base is not None:
            self.overwrite_base = overwrite_base

        new_config = self.get_config()
        config_to_copy = copy.deepcopy(new_config['layers'][layer_number])
        unit_count = config_to_copy['config']['units']
        config_to_copy['config']['activation'] = activation
        config_to_copy['config']['kernel_initializer'] = {'class_name': 'Identity', 'config': {'gain': 1.0}}
        config_to_copy['config']['use_bias'] = use_bias
        config_to_copy['config']['bias_initializer'] = {'class_name': 'Zeros', 'config': {}}
        config_to_copy['config']['name'] = 'dense_' + str(self.layer_name)
        new_config['layers'].insert(layer_number + 1, config_to_copy)

        new_model = keras.Sequential.from_config(new_config)

        w0 = np.identity(unit_count)
        b0 = np.zeros((unit_count,))

        # concat, set weights
        new_weights = []
        for idx in range(layer_number - self.index_shift):
            new_weights += [self.base_weights[idx]] 
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_weights += [w0] 
        if use_bias:
            new_weights += [b0]

        for idx in range(layer_number - self.index_shift, len(self.base_weights)):
            new_weights += [self.base_weights[idx]]
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_model.set_weights(new_weights)

        return self.return_compiled_model(new_model)


    def RemoveLayer(self, layer_number, overwrite_base=None):
        '''
        Removes the specified layer (starting from 1, can't remove input layer)
        '''
        check_ok, layer_number = self.perform_checks(layer_number)
        if not check_ok:
            return None

        if overwrite_base is not None:
            self.overwrite_base = overwrite_base

        new_config = self.get_config()
        
        use_bias_this = new_config['layers'][layer_number]['config']['use_bias']
        use_bias_next = new_config['layers'][layer_number + 1]['config']['use_bias'] 
        use_bias = use_bias_this or use_bias_next
        new_config['layers'][layer_number + 1]['config']['use_bias'] = use_bias

        del new_config['layers'][layer_number] # do not count the input layer!

        new_model = keras.Sequential.from_config(new_config)

        # generate adapted weights
        w0 = self.base_weights[layer_number - 1 - self.index_shift]
        w1 = self.base_weights[layer_number - self.index_shift]
        w_new = w0 @ w1
        
        b_new = 0
        if use_bias_this:
            b0 = self.base_biases[layer_number - 1 - self.index_shift]
            b_new += b0 @ w1
        if use_bias_next:
            b1 = self.base_biases[layer_number - self.index_shift]
            b_new += b1
                
        # concat, set weights
        new_weights = []
        for idx in range(layer_number-1 - self.index_shift):
            new_weights += [self.base_weights[idx]]
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_weights += [w_new]
        if use_bias:
            new_weights += [b_new]

        for idx in range(layer_number + 1 - self.index_shift, len(self.base_weights)):
            new_weights += [self.base_weights[idx]]
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_model.set_weights(new_weights)

        return self.return_compiled_model(new_model)

    def PerformSVD(self, layer_number, type='truncated', neurons_to_remove=1, overwrite_base=None, activation=None):
        if type == 'truncated':
            return self.PerformTruncatedSVD(layer_number, neurons_to_remove=neurons_to_remove, overwrite_base=overwrite_base, activation=activation)
        elif type == 'oneLayer':
            return self.PerformOneLayerSVD(layer_number, neurons_to_remove=neurons_to_remove, overwrite_base=overwrite_base)
        else:
            print("Method not found")
            return None


    def PerformTruncatedSVD(self, layer_number, neurons_to_remove=1, activation=None, use_bias=False, last_layer_allowed=False, overwrite_base=None):
        '''
        Builds a model with one more layer, but reduced connections
        '''
        check_ok, layer_number = self.perform_checks(layer_number)
        if not check_ok:
            return None

        if overwrite_base is not None:
            self.overwrite_base = overwrite_base

        new_config = self.get_config()
        config_to_copy = copy.deepcopy(new_config['layers'][layer_number])
        desired_units = config_to_copy['config']['units'] - neurons_to_remove
        use_bias_next = new_config['layers'][layer_number]['config']['use_bias']

        # Generate new weights before generating model, as unit number can still vary
        w = self.base_weights[layer_number - 1 - self.index_shift]
        u, s, vh = np.linalg.svd(w, full_matrices=True, compute_uv=True)
        
        # s will most likely have full length, even if the rank is smaller (due to machine precision)
        # check for this in any case!
        desired_units = np.minimum(desired_units, len(s))
        s = s[:desired_units]
        u = u[:,:desired_units]
        vh = vh[:desired_units, :]
        
        # build model with required number of units
        config_to_copy['config']['units'] = desired_units
        config_to_copy['config']['activation'] = activation
        config_to_copy['config']['use_bias'] = use_bias
        config_to_copy['config']['bias_initializer'] = None
        config_to_copy['config']['name'] = 'dense_' + str(self.layer_name)
        self.layer_name += 1
        new_config['layers'].insert(layer_number, config_to_copy)

        new_model = keras.Sequential.from_config(new_config)

        # concat, set weights
        new_weights = []
        for idx in range(layer_number-1 - self.index_shift):
            new_weights += [self.base_weights[idx]]
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_weights += [u * s]
        if use_bias:
            new_weights += [np.zeros(desired_units)]

        new_weights += [vh]
        if use_bias_next:
            new_weights += [self.base_biases[layer_number - 1 - self.index_shift]]
        
        for idx in range(layer_number - self.index_shift, len(self.base_weights)):
            new_weights += [self.base_weights[idx]]
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_model.set_weights(new_weights)

        return self.return_compiled_model(new_model)


    def ChangeActivation(self, layer_number, activation, overwrite_base=None):
        '''
        Builds a model with the given activation function in the specified layer
        Weights are not adapted
        '''
        check_ok, layer_number = self.perform_checks(layer_number)
        if not check_ok:
            return None

        if overwrite_base is not None:
            self.overwrite_base = overwrite_base

        new_config = self.get_config()
        new_config['layers'][layer_number]['config']['activation'] = activation

        new_model = keras.Sequential.from_config(new_config)
        new_model.set_weights(self.all_weights)

        return self.return_compiled_model(new_model)

    def PerformOneLayerSVD(self, layer_number, neurons_to_remove=1, overwrite_base=None):
        '''
        Builds a model with less units in the specified layer by using SVD to project to
        a lower rank subspace. The backwards projection is pulled into the next layer,
        so a distortion occurs due to the activation function.
        '''

        check_ok, layer_number = self.perform_checks(layer_number, neurons_to_remove=neurons_to_remove)
        if not check_ok:
            return None
        
        if overwrite_base is not None:
            self.overwrite_base = overwrite_base

        new_config = self.get_config()
        desired_units = new_config['layers'][layer_number]['config']['units'] - neurons_to_remove
        use_bias_this = new_config['layers'][layer_number]['config']['use_bias']
        use_bias_next = new_config['layers'][layer_number+1]['config']['use_bias']

        # Generate new weights before generating model, as unit number can still vary
        w0 = self.base_weights[layer_number - 1 - self.index_shift]
        w1 = self.base_weights[layer_number - self.index_shift]
        u, s, vh = np.linalg.svd(w0, full_matrices=True, compute_uv=True)

        # s will most likely have full length, even if the rank is smaller (due to machine precision)
        # check for this in any case!
        desired_units = np.minimum(desired_units, len(s))
        s = s[:desired_units]
        u = u[:,:desired_units]
        
        # build model with required number of units
        new_config['layers'][layer_number]['config']['units'] = desired_units
        new_model = keras.Sequential.from_config(new_config)

        # concat, set weights
        new_weights = []
        for idx in range(layer_number-1 - self.index_shift):
            new_weights += [self.base_weights[idx]]
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_weights += [u * s]
        if use_bias_this:
            b_new = vh.T @ self.base_biases[layer_number - 1 - self.index_shift]
            new_weights += [b_new[:desired_units]]

        w1_new = vh @ w1
        new_weights += [w1_new[:desired_units, :]]
        if use_bias_next:
            new_weights += [self.base_biases[layer_number - self.index_shift]]

        for idx in range(layer_number + 1 - self.index_shift, len(self.base_weights)):
            new_weights += [self.base_weights[idx]]
            bias = self.base_biases[idx]
            if bias is not None:
                new_weights += [bias]

        new_model.set_weights(new_weights)

        return self.return_compiled_model(new_model)