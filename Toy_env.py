""" 
The environment simulates the chain problem
"""

import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

from deer.base_classes import Environment

class MyEnv(Environment):
    
    def __init__(self, n_states=5):
        """ Initialize environment.
        """
        self.n_states = n_states
        self.n_actions = 2
        self._last_ponctual_observation = 0
                
    def reset(self, mode):
        """ Resets the environment for a new episode.
        """
        self._last_ponctual_observation = 0
        return self.observe()

    def act(self, action):
        """ Performs one time-step within the environment

        Parameters
        -----------
        action : int
            Integer in [0, ..., N_A-1] where N_A is the number of actions given by self.nActions()

        Returns
        -------
        reward: float
        """        
        if action == 1:
            if self._last_ponctual_observation == 0:
                return 0.2
            else:
                self._last_ponctual_observation = 0
                return 0.
        else:
            if self._last_ponctual_observation == self.n_states - 1:
                return 1
            else:
                self._last_ponctual_observation += 1
                return 0.
    
    def inputDimensions(self):
        return [(1,)]           # state is simply defined as one scalar (without history)

    def nActions(self):
        return 2                # The environment allows two different actions to be taken at each time step

    def inTerminalState(self):
        return False

    def observe(self):
        return np.array([self._last_ponctual_observation])



def main():
    # Can be used for debug purposes
    myenv = MyEnv()

    print (myenv.observe())
    
if __name__ == "__main__":
    main()
