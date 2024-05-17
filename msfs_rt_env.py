from gymnasium import spaces
from default_env import DefaultEnv
from rtgym import RealTimeGymInterface

import logging

logging.getLogger("root").setLevel(40)


class RealTimeFlightSimInterface(RealTimeGymInterface):
    """The Default MSFS interface class. inherit to create a custom envirnment
    Overide methods if nescesary.
    """

    def __init__(self, env=DefaultEnv()):
        """Initalise the interface with a chosen environment.
        Call this method from the custom child class and pass your custom MSFS env class.

        Kwargs:
            env (DefaultInerface): The MSFS environment class you wish to interface with.
        """

        self.env = env

    def get_observation_space(self) -> spaces.Tuple:
        """Return the observation space of the environment

        Returns:
            spaces.Tuple: rtgym requires the observation space to be a spaces.Tuple.
        """

        return self.env.get_observation_space()

    def get_action_space(self):
        """Return the action space of the environment"""

        return self.env.get_action_space()

    def get_default_action(self):
        """Return the observation space of the environment"""

        return self.env.get_default_action()

    def send_control(self, control):
        """Send a control to MSFS via the environment

        Args:
            control: The control input to MSFS
        """

        self.env.control(control)

    def reset(self, seed=None, options=None):
        """Reset the environment and return an observation

        Kwargs:
            seed: None
            options: None

        Returns:
            obs: Observation from the environemnt
            info: empty
        """

        self.env.reset_env()

        obs = self.env.observation()

        return obs, {}

    def get_obs_rew_terminated_info(self):
        """Return an observation, reward, terminated, info

        Returns:
            obs: Observation from the environemnt
            reward: Reward value from the environment
            term: Result of the termination condition
            info: empty
        """

        obs = self.env.observation()

        rew = self.env.reward(obs)

        term = self.env.terminate()

        return obs, rew, term, {}

    def wait(self):
        """Do nothing"""
        pass
