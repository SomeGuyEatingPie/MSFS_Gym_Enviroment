from simconnect import SimConnect
from gymnasium import spaces
import numpy as np


class DefaultEnv:
    """The Default MSFS environment class. Inherit to create a custom environment"""

    def __init__(self):
        """Either call this method from the custom child class,
        or write your own code to connect to SimConnect"""

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

    def get_observation_space() -> spaces.Tuple:
        """Overide me!
        Return the observation space of the environment

        Returns:
            spaces.Tuple: rtgym requires the observation space to be a spaces.Tuple.
        """

        raise NotImplementedError

    def get_action_space() -> spaces.Box:
        """Overide me!
        Return the observation space of the environment
        """

        raise NotImplementedError

    def get_default_action() -> np.array:
        """Overide me!
        Return the default action of the environment
        """

        raise NotImplementedError

    def observation():
        """Overide me!
        Return an observation
        """

        raise NotImplementedError

    def control():
        """Overide me!
        Send a control input to MSFS
        """

        raise NotImplementedError

    def reset_env():
        """Overide me!
        Reset the environment
        """

        raise NotImplementedError

    def reward(obs):
        """Overide me!
        Return the agent's reward

        Args:
            obs: the most recent observation
        """

        raise NotImplementedError

    def terminate() -> bool:
        """Overide me!
        Return True if the episode should terminate
        """

        raise NotImplementedError
