# Copyright (c) 2021 Schiessler et al.

# store model modifications
import numpy as np
import copy

class Modification:
    def __init__(self, model, id):
        '''
        Initializes a new instance of the Modification class and assigns the provided ID
        '''
        # set known properties
        self.model = model
        self.id = id

        # initialize unknown properties - need to be updated from outside
        self.parent = -1
        self.description = ''
        self.is_random = False
        self.history = {}
        self.score = -1
        self.parameter_increase = 0.0
        self.log = []

    def generate_descendant(self, desc_model, desc_id):
        '''
        Generates a new modification instance that is a denscendant of the current instance.
        Current instance will be set as parent.
        History will be copied.
        '''
        descendant = Modification(desc_model, desc_id)
        descendant.parent = self.id
        descendant.history = copy.deepcopy(self.history)
        descendant.description = self.description
        return descendant

    def calculate_score(self, new_history, scoring_key, append=True):
        '''
        Scores the modification based on the passed history object and the stored history,
        using the provided scoring key.
        If append is set to True, the new history will be appended to the stored history.
        '''
        old_value = 1
        
        for key in new_history.keys():
            if key not in self.history.keys():
                self.history[key] = []
            elif key == scoring_key:
                old_value = self.history[scoring_key][-1]
        self.log.append('old_value:')
        self.log.append(old_value)
        new_value = new_history[scoring_key][-1]
        self.log.append('new_value')
        self.log.append(new_value)
        if True:
            for key in new_history.keys():
                for value in new_history[key]:
                    self.history[key].append(value)
        
        self.score = new_value + np.exp(new_value - old_value)*np.exp(- self.parameter_increase)
        self.log.append('Parameter_increase')
        self.log.append(self.parameter_increase)
        self.log.append('score')
        self.log.append(self.score)

    @property
    def singular_values(self):
        '''
        Computes singular values for the weights of the stored model.
        Layers without weights are included as None
        '''
        singular_values = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                w = weights[0]
                if np.isnan(w).any():
                    singular_values.append(None)
                    continue
                singular_values.append(np.linalg.svd(weights[0], compute_uv=False))
            else:
                singular_values.append(None)
        return singular_values

    def evaluate(self, *args, **kwargs):
        '''
        Pass through function to access model.evaluate
        '''
        return self.model.evaluate(*args, **kwargs)

    def summary(self, *args, **kwargs):
        '''
        Pass through function to access model.summary
        '''
        return self.model.summary(*args, **kwargs)
