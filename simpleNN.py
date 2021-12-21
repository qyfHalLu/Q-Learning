import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Dense, Flatten, concatenate, Activation, Reshape, Permute

class simpleNN():
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
        Set the random seed.
    action_as_input : Boolean
        Whether the action is given as input or as output
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state, action_as_input=False):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._action_as_input=action_as_input

    def _buildDQN(self):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        inputs=[]

        for i, dim in enumerate(self._input_dimensions):
            # - observation[i] is a SCALAR -
            input = Input(shape=(dim[0],))
            inputs.append(input)
                    
        if self._action_as_input==True:
            if isinstance(self._n_actions,int):
                print("Error, env.nActions() must be a continuous set when using actions as inputs in the NN")
            else:
                input = Input(shape=(len(self._n_actions),))
                inputs.append(input)
        
        x = concatenate(inputs)
        
        # we stack a deep fully-connected network on top
        x = Dense(10, activation='relu')(x)
        
        if self._action_as_input==False:
            if isinstance(self._n_actions,int):
                out = Dense(self._n_actions)(x)
            else:
                out = Dense(len(self._n_actions))(x)
        else:
            out = Dense(1)(x)

        model = Model(inputs=inputs, outputs=out)
        layers=model.layers
        
        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]
        
        if self._action_as_input==True:
            return model, params, inputs
        else:
            return model, params