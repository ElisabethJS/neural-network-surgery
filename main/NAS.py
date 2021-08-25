# Copyright (c) 2021 Schiessler et al.

import os
import sys
import uuid
import numpy as np
import copy
from enum import Enum
import tensorflow as tf
from tensorflow import keras
import Modification
import ModelAdaptations

keras_index_shift = 1 if keras.__version__ >= '2.3.0' else 0 # Required bc. keras internal configuration dictionary for models always includes an
# input layer starting from version 2.3.0, that is not included in the list of model.layers 

class NasTrainer:

    # Configuration settings for the Surgeon
    config = {
        'epoch_step': 10,                       # number of epochs in each cycle
        'initial_epochs': 10,                   # model is initially trained for this number of epochs before optimization starts
        'include_random_modification': True,    # shall a random modification be added to the competition in each optimization step
        'max_epochs': 100,                      # number of epochs of training that the winning model will receive
        'max_retries_per_epoch': 3,             # max. number of re-tries for the competition in each optimization step
        'parallel_branches': 2,                 # max. number of concurrent modification that are retained simultaneously
        'retraining_batches': 25,               # number of batches for retraining. each modification will receive a multiple of this amount, determined by its modification type
        'sv_treshold_abs': 3e-1,                # relative singular value threshold, used for decisions about pruning
        'sv_treshold_rel': 5e-3                 # absolute singular value threshold, used for decisions about pruning
    }

    def __init__(self, data, compiler_information, scoring_criterion='val_accuracy'):
        '''
        Initializer for the neural architecture search class.
        
        Parameters:
        data:                   tuple containing training and test set
        compiler_information:   dictionary containing entries for loss function, training metrics as well as optimizer function. These are used to compile the tf model
        scoring_criterion:      criterion to be used in branch scoring
        '''
        self.loss = compiler_information['loss']
        self.metrics = compiler_information['metrics']
        self.optimizer = compiler_information['optimizer']

        ds_train, ds_valid = data
        self.ds_train = ds_train
        self.ds_valid = ds_valid

        self.train_batch_cnt = len(self.ds_train)
        uuid4 = uuid.uuid4()
        self.tmp_model_path = os.path.join('.','tmp', 'model_{}.h5'.format(uuid4))
        self.scoring_criterion = scoring_criterion

    def run(self, model, random_only=False, verbose=1):
        '''
        Runs the neural architecture search algorithm. Logs are re-set to empty each time this function is called
        
        Parameters:
        model:          The original tensorflow model. Needs to be sequential, start with a flatten layer, and contain only dense layers otherwise
        random_only:    Random walk mode. If set to True, only one new modification will be randomly created in each optimization step.
        verbose:        Verbosity level for output logs. 2..include all messages, 1..print modification information only, 0..print nothing
        
        Returns:
            winning modification (of type Modification, includes the fully trained model), including genetic information about the winning model
        '''
        self.__id_counter = 0
        self.__epoch_counter = 0
        self.__all_potential_modification_candidates = {}
        self.__all_trained_modifications = {}
        self.__retrain_amount_per_modification = {}
        self.__verbose = verbose

        self.log = []

        result = self.__perform_surgery(model, random_only)

        self.__cleanup()

        return result

    def __perform_surgery(self, model, random_only):
        '''
        Main algorithm function.
        '''
        self.__print('Started run', 1)
        epoch_step = self.config['epoch_step']
        
        # initialize with provided model
        current_branches = [self.__parse_initial_model(model)]
        current_best_id = current_branches[0].id
        current_best_value = 0
        last_best_value = current_branches[0].history[self.scoring_criterion][-1]

        # perform NAS until max_epochs has been reached
        while self.__epoch_counter < self.config['max_epochs']:
            self.__epoch_counter +=1
                        
            self.__print('Current epoch: ' + str(self.__epoch_counter) + ' to ' + str(self.__epoch_counter + epoch_step - 1), 1)

            for modification in current_branches:
                self.__create_potential_modification_candidates(modification)

            self.__epoch_counter += epoch_step - 1
            
            current_best_id = -1
            current_best_value = 0
            new_branches = []
            tries = 0
            while True:
                tries += 1
                self.__print('..try ' + str(tries), 1)
                modifications = self.__select_from_candidates(current_branches, random_only)
                self.__print(str(len(modifications)) + ' modifications selected', 2)
                
                if len(modifications) == 0:
                    # no more candidates selectable
                    last_best = self.__get_last_best(current_best_id)
                    if last_best is not None:
                        new_branches.append(last_best)
                    break

                modifications = self.__retrain_and_select_new_branches(modifications, random_only)
            
                for modification in modifications:
                    retrain_amount = self.train_batch_cnt - self.__retrain_amount_per_modification[modification.id]
                    self.__perform_training_step(modification, number_of_epochs=epoch_step - 1, 
                                                 training_batches=retrain_amount, 
                                                 update_history=True)
                    
                    current_scoring_value = modification.history[self.scoring_criterion][-1]
                    
                    if current_scoring_value > current_best_value:
                        current_best_id = modification.id
                        current_best_value = current_scoring_value

                    if random_only or current_scoring_value >= last_best_value or self.config['max_retries_per_epoch'] == 1:
                        new_branches.append(modification)

                if len(new_branches) > 0:
                    self.__print(str(len(new_branches)) + ' modifications kept in current branches', 2)
                    break

                if self.config['max_retries_per_epoch'] >= 0 and self.config['max_retries_per_epoch'] <= tries:
                    last_best = self.__get_last_best(current_best_id)
                    if last_best is not None:
                        new_branches.append(last_best)
                    break
            
            if len(new_branches) > 0:
                current_branches = new_branches
                last_best_value = current_best_value
            else:
                for modification in current_branches:
                    self.__perform_training_step(modification, number_of_epochs=epoch_step, training_batches=[], update_history=True)

        # select and return highest scoring modification from current candidates
        winning_modification = self.__evaluate_modifications(current_branches, random_only=random_only, desired_number=1)
        if len(winning_modification) == 0:
            winning_modification = [self.__get_last_best(current_best_id)]
        return winning_modification[0]

    def __get_last_best(self, best_id):
        '''
        Called if none of the newly generated modifications manages to score better than the last best scoring one. It returns the
        previously best scoring modification
        '''
        if best_id == -1:
            return None
        last_best = self.__all_trained_modifications[best_id]
        self.__print('kept last best', 2)
        return last_best

    def __print(self, message, verbose_bar=1):
        '''
        Printing function that appends messages to the log. The verbosity parameter is used to determine which messages are included.
        '''
        if verbose_bar > 1:
            message = '  ' + message
        self.log.append(message)
        if self.__verbose >= verbose_bar:
            print(message)

    def __retrain_and_select_new_branches(self, modifications, random_only):
        '''
        Each modification receives retraining as specified by its type, then selects the top scoring modifications
        '''
        for modification in modifications:
            retrain_amount = self.__retrain_amount_per_modification[modification.id]
            self.__perform_training_step(modification, number_of_epochs=0, 
                                         training_batches=retrain_amount, 
                                         update_history=False)
                
        return self.__evaluate_modifications(modifications, random_only=random_only)

    def __store_modification(self, modification):
        '''
        Appends the modification to the dictionary containing all modifications that have been trained at any point
        '''
        self.__all_trained_modifications[modification.id] = modification

    def __parse_initial_model(self, model):
        '''
        Turns the initial model into a modification candidate, sets initial values for various internals
        '''
        self.__print('Initial training', 1)
        # assign first modification object and train for initial number of epochs
        initial_modification = self.__assign_modification(model, 'B', parent=None)
        self.__epoch_counter += self.config['initial_epochs']

        self.__perform_training_step(initial_modification, number_of_epochs=self.config['initial_epochs'], update_history=True)      
        
        return initial_modification

    def __assign_modification(self, model, description, parent=None):
        '''
        Generates a new modification object for the given model.
        If a parent is specified, it is used to generate a descendant.
        '''
        model.save(self.tmp_model_path)
        new_model = tf.keras.models.load_model(self.tmp_model_path)
        new_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if parent is None:
            m = Modification.Modification(new_model, self.__id_counter)
        else:
            m = parent.generate_descendant(new_model, self.__id_counter)
        self.__id_counter += 1

        m.description += description
        return m

    def __perform_training_step(self, modification, number_of_epochs=0, training_batches=0, update_history=True):
        '''
        Trains the provided modification's model. Either number_of_epochs or training_batches must be provided.
        In case both are provided, both are used.
        If update_history is True, the modification's history will be updated when calculating its score.
        '''
        
        if training_batches == 0 and number_of_epochs == 0:
            return
        
        if training_batches > 0:  
            history = modification.model.fit(self.ds_train.take(training_batches), epochs=1, verbose=0,
                                             validation_data=self.ds_valid)
            self.__calculate_score(modification, history, update_history=update_history)

        if number_of_epochs > 0:
            history = modification.model.fit(self.ds_train, epochs=number_of_epochs, verbose=0,
                                             validation_data=self.ds_valid)
            self.__calculate_score(modification, history, update_history=update_history)
            
        self.__store_modification(modification)

    def __calculate_score(self, modification, history, update_history=True):        
        '''
        Calls the modification's calculate_score method with the correct values.
        '''
        modification.calculate_score(history.history, self.scoring_criterion, append=update_history)
        
    def __create_potential_modification_candidates(self, base_modification):
        '''
        Analyses the base_modification and creates all potential modification candidates.
        These are stored in an internal dictionary by base_modification.id.
        '''
        
        svs = base_modification.singular_values

        candidates = []
        desc_0 = ' ~ EP ' + str(self.__epoch_counter)
        base_trainable_variables = self.__count_trainable_variables(base_modification)
        
        # parent is always added
        candidates.append(modificationCandidate(probability=1.0, kwargs={}, creator_func=lambda ma, kwargs: ma.Identity(**kwargs),
                                                description= desc_0 + ' B', retrain_amount=1, 
                                                parameter_increase=1.0, parent_id=base_modification.id,
                                                modification_type=modificationType.BASE))
        
        layer_count = len(base_modification.model.layers)
        for i in range(2, layer_count):
            # Values
            config = base_modification.model.get_config()['layers'][i]['config']
            neuron_count = config['units']
            sv = svs[i]
            layer_name = config['name']
            desc_1 = ' ' + layer_name
            bias_multiplicator = 1 if config['use_bias'] else 0
            layer_weight_shape = base_modification.model.layers[i-1].get_weights()[0].shape

            # Remove 'illegal' layers
            if neuron_count == 0 or sv is None:
                new_param_count = base_trainable_variables - layer_weight_shape[0]*layer_weight_shape[1] - bias_multiplicator*neuron_count
                param_increase = new_param_count / base_trainable_variables
                candidates.append(modificationCandidate(probability=1.0, kwargs={'layer_number': i}, creator_func=lambda ma, kwargs: ma.RemoveLayer(**kwargs),
                                                        description= desc_0 + ' RL' + desc_1, retrain_amount=10,
                                                        parameter_increase=param_increase, parent_id=base_modification.id,
                                                        modification_type=modificationType.REMOVE_LAYER))
                continue
            
            # Counters
            removable_neurons = len([s for s in sv if (s < self.config['sv_treshold_abs']) or (s < self.config['sv_treshold_rel']*sv[0])])
            add_amount = np.maximum(2, neuron_count//10)
            remove_amount = np.maximum(1, removable_neurons)
            remove_amount = np.minimum(removable_neurons, neuron_count-1)

            # Probabilities
            p_add_neuron = 1 - removable_neurons/neuron_count
            p_add_layer = p_add_neuron*(i/(layer_count-1)) # rises with layer number
            p_remove_neuron = removable_neurons/neuron_count
            p_remove_layer = p_remove_neuron*(i/(layer_count-1))

            # Assign candidates
            
            kwargs = {'layer_number': i}
            
            new_param_count = base_trainable_variables + neuron_count * (neuron_count + 1)
            param_increase = new_param_count / base_trainable_variables
            candidates.append(modificationCandidate(probability=p_add_layer, kwargs=copy.copy(kwargs), creator_func=lambda ma, kwargs: ma.AddLayer(**kwargs),
                                                    description=desc_0 + ' AL' + desc_1, retrain_amount=1,
                                                    parameter_increase=param_increase, parent_id=base_modification.id,
                                                    modification_type=modificationType.ADD_LAYER))
            
            new_param_count = base_trainable_variables - layer_weight_shape[0]*layer_weight_shape[1] - bias_multiplicator*neuron_count           
            param_increase = new_param_count / base_trainable_variables
            candidates.append(modificationCandidate(probability=p_remove_layer, kwargs=copy.copy(kwargs), creator_func=lambda ma, kwargs: ma.RemoveLayer(**kwargs),
                                                    description=desc_0 + ' RL' + desc_1, retrain_amount=10,
                                                    parameter_increase=param_increase, parent_id=base_modification.id,
                                                    modification_type=modificationType.REMOVE_LAYER))
            
            kwargs['neurons_to_add'] = add_amount
            new_param_count = base_trainable_variables + (layer_weight_shape[0] + 1) * add_amount
            param_increase = new_param_count / base_trainable_variables
            candidates.append(modificationCandidate(probability=p_add_neuron, kwargs=copy.copy(kwargs), creator_func=lambda ma, kwargs: ma.AddNeuron(**kwargs),
                                                    description=desc_0 + ' AN' + str(add_amount) + desc_1, retrain_amount=1, 
                                                    parameter_increase=param_increase, parent_id=base_modification.id,
                                                    modification_type=modificationType.ADD_NEURON))
            add_amount = add_amount*10
            kwargs['neurons_to_add'] = add_amount
            new_param_count = base_trainable_variables + (layer_weight_shape[0] + 1) * add_amount
            param_increase = new_param_count / base_trainable_variables
            candidates.append(modificationCandidate(probability=p_add_neuron, kwargs=copy.copy(kwargs), creator_func=lambda ma, kwargs: ma.AddNeuron(**kwargs),
                                                    description=desc_0 + ' AN' + str(add_amount) + desc_1, retrain_amount=1, 
                                                    parameter_increase=param_increase, parent_id=base_modification.id,
                                                    modification_type=modificationType.ADD_NEURON))

            del kwargs['neurons_to_add']
            kwargs['neurons_to_remove'] = remove_amount          
            new_param_count = base_trainable_variables - (layer_weight_shape[0] + bias_multiplicator) * remove_amount
            param_increase = new_param_count / base_trainable_variables
            candidates.append(modificationCandidate(probability=p_remove_neuron, kwargs=copy.copy(kwargs), creator_func=lambda ma, kwargs: ma.PerformOneLayerSVD(**kwargs),
                                                    description=desc_0 + ' OL' + str(remove_amount) + desc_1, retrain_amount=10, 
                                                    parameter_increase=param_increase, parent_id=base_modification.id,
                                                    modification_type=modificationType.REMOVE_NEURON))

            new_param_count = base_trainable_variables - (layer_weight_shape[0]*layer_weight_shape[1]) 
            new_param_count += (layer_weight_shape[1] - remove_amount)*(layer_weight_shape[0] + layer_weight_shape[1])
            param_increase = new_param_count / base_trainable_variables
            candidates.append(modificationCandidate(probability=p_remove_neuron, kwargs=copy.copy(kwargs), creator_func=lambda ma, kwargs: ma.PerformTruncatedSVD(**kwargs),
                                                    description=desc_0 + ' T' + str(remove_amount) + desc_1, retrain_amount=1, 
                                                    parameter_increase=param_increase, parent_id=base_modification.id,
                                                    modification_type=modificationType.TRUNCATE))
        
        self.__print(str(len(candidates)) + ' candidates created', 2)
        self.__all_potential_modification_candidates[base_modification.id] = candidates

    def __count_trainable_variables(self, base_modification):
        '''
        Gets the number of trainable variables of the given modification's model.
        '''
        tvs = base_modification.model.trainable_variables
        count = 0
        for tv in tvs:
            shape = tv.shape
            if len(shape) == 1:
                count += shape[0]
            else:
                count += shape[0]*shape[1]
        
        return count

    
    def __select_from_candidates(self, base_modifications, random_only):
        '''
        For each base modification goes through all potential modification candidates that have not yet been trained,
        and selects the desired number
        '''
        candidates = []

        for base_modification in base_modifications:
            candidates += [c for c in self.__all_potential_modification_candidates[base_modification.id] if c.trained == False]
            
        self.__print(str(len(candidates)) + ' candidates found', 2)
        if len(candidates) == 0:
            return []
        
        result = []
        not_chosen = []
        if not random_only:
            for mtype in modificationType:
                cands = [c for c in candidates if c.modification_type == mtype]
                if len(cands) == 0:
                    continue

                cands.sort(key=lambda x: (x.probability, x.parameter_increase), reverse=True)
                idx = 0
                while idx < len(cands):
                    new_modification = self.__parse_modification_candidate(cands[idx])
                    idx += 1
                    if new_modification is not None:
                        result.append(new_modification)
                        if mtype != modificationType.BASE:
                            break
                if idx < len(cands):
                    not_chosen += cands[idx:]
        
        # add random member - if not all used anyways
        if self.config['include_random_modification'] and len(not_chosen) > 0:
            candidate = np.random.default_rng().choice(not_chosen)
            candidate.description += '**' # mark random mutations in permutation list
            new_modification = self.__parse_modification_candidate(candidate)
            if new_modification is not None:
                new_modification.is_random = True
                result.append(new_modification)

        # return result
        return result
        

    def __evaluate_modifications(self, modifications, random_only=False, desired_number=None):
        '''
        Evaluates which modifications have valid weights, and returns the desired number of modifications ranked by sorting criterion
        '''
        allowed_modifications = [] # Exclude modifications that have degenerate weights
        for modification in modifications:
            include = True
            for weight in modification.model.get_weights():
                if include and np.isnan(weight).any():
                    include = False
                    self.__print(modification.description + ' not allowed', 2)
                    self.__print('ID: ' + str(modification.id), 2)
            if include:
                allowed_modifications.append(modification)
        self.__print(str(len(allowed_modifications)) + ' modifications allowed', 2)

        if random_only:
            idx = self.__rng.choice(len(allowed_modifications))
            return [allowed_modifications[idx]]

        if desired_number is None:
            desired_number = self.config['parallel_branches']

        if desired_number >= len(allowed_modifications):
            return allowed_modifications

        result = allowed_modifications[:desired_number]
        self.__sort_modifications(result)
        for modification in allowed_modifications[desired_number:]:
            if self.__sorting_criterion(modification) > self.__sorting_criterion(result[-1]):
                result.append(modification)
                self.__sort_modifications(result)
                result = result[:desired_number]

        return result


    def __sort_modifications(self, modifications):
        '''
        Ranks modifications by score
        '''
        modifications.sort(reverse=True, key=self.__sorting_criterion)

    def __sorting_criterion(self, modification):
        '''
        Returns the score of a modification
        '''
        return modification.score

    def __parse_modification_candidate(self, modification_candidate):
        '''
        Generates a modification object from a given modification candidate
        '''
        modification_candidate.trained = True
        parent = self.__all_trained_modifications[modification_candidate.parent_id]
        ma = ModelAdaptations.ModelAdapter(parent.model, loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, overwrite_base=False, index_shift = keras_index_shift)
        new_model = modification_candidate.creator_func(ma, modification_candidate.kwargs)
        if new_model is None:
            return
        new_modification = self.__assign_modification(new_model, modification_candidate.description, parent=parent)
        new_modification.parameter_increase = modification_candidate.parameter_increase
        self.__retrain_amount_per_modification[new_modification.id] = modification_candidate.retrain_amount
        return new_modification

    def __cleanup(self):
        '''
        Removes the temporary model file
        '''
        os.remove(self.tmp_model_path)

# class that contains all possible choices of modification types
class modificationType(Enum):
    UNSPECIFIED = -1
    BASE = 0
    ADD_NEURON = 1
    ADD_LAYER = 2
    REMOVE_NEURON = 3
    REMOVE_LAYER = 4
    TRUNCATE = 5


# class that stores all information about potential modification candidates
class modificationCandidate:
    def __init__(self, probability=0, kwargs={}, creator_func=None, description='', retrain_amount=0,
                 parameter_increase=0.0, parent_id=-1, modification_type=modificationType.UNSPECIFIED):
        self.creator_func = creator_func
        self.description = description
        self.kwargs = kwargs
        self.modification_type = modification_type
        self.parameter_increase = parameter_increase
        self.parent_id = parent_id
        self.probability = probability
        self.retrain_amount = retrain_amount
        self.trained = False
