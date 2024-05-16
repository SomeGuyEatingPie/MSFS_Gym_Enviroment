from gymnasium import spaces
from default_env import DefaultEnv
from rtgym import RealTimeGymInterface

import logging
logging.getLogger("root").setLevel(40)

class RealTimeFlightSimInterface(RealTimeGymInterface):

    def __init__(self, env = DefaultEnv()):
        
        self.env = env

    def get_observation_space(self) -> spaces.Tuple:

       return self.env.get_observation_space()

    def get_action_space(self) -> spaces.Box:

        return  self.env.get_action_space()

    def get_default_action(self):

        return self.env.get_default_action()

    def send_control(self, control):

        self.env.control(control)

    def reset(self, seed=None, options=None):

        self.env.reset_env()

        obs = self.env.observation()
        
        return obs, {}

    def get_obs_rew_terminated_info(self):
        
        obs = self.env.observation()
        
        rew = self.env.reward(obs)

        term = self.env.terminate()
        
        return obs, rew, term, {}

    def wait(self):
        pass



