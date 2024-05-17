from simconnect import SimConnect
from gymnasium import spaces
import numpy as np


class DefaultEnv:

    def __init__(self):

        f = False
        while f == False:
            try:
                self.sc = SimConnect(name="RTGym")
                f = True
            except:
                print(
                    "Error, cannot connect to simconnect. Press ENTER to try again..."
                )
                input()

    def get_observation_space(self) -> spaces.Tuple:

        raise NotImplementedError

    def get_action_space(self) -> spaces.Box:

        raise NotImplementedError

    def get_default_action(self) -> np.array:

        raise NotImplementedError

    def observation(self):

        raise NotImplementedError

    def control(self, controlInput):

        raise NotImplementedError

    def reset_env(self, seed=None, options=None):

        raise NotImplementedError

    def reward(self, observation):

        raise NotImplementedError

    def terminate(self, observation):

        raise NotImplementedError
