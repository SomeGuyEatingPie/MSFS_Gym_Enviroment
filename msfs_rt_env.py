import gymnasium as gym
from gymnasium import spaces
from simconnect_env import MSFS
import ray
from ray.rllib.algorithms.sac import SAC
from ray.tune.logger import pretty_print
from ray.rllib.utils import check_env
import numpy as np
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT

import logging
logging.getLogger("root").setLevel(40)

class MyRealTimeInterface(RealTimeGymInterface):

    def __init__(self):
        
        self.env = MSFS()

    def get_observation_space(self) -> spaces.Tuple:

        heading = spaces.Box(low=-360.0, high=360.0, shape=(1,))
        bank = spaces.Box(low=-360.0, high=360.0, shape=(1,))
        pitch = spaces.Box(low=-360.0, high=360.0, shape=(1,))
        turnCoord= spaces.Box(low=-127.0, high=127, shape=(1,))
        airspeed = spaces.Box(low=-10.0, high=np.inf, shape=(1,))
        vario = spaces.Box(low=-np.inf, high=100.0, shape=(1,))
        trueAlt = spaces.Box(low=0.0, high=13000.0, shape=(1,))
        alt = spaces.Box(low=0.0, high=13000.0, shape=(1,))
        pos = spaces.Box(low=-180.0, high=180.0, shape=(2,))

        return spaces.Tuple((heading, bank, pitch, turnCoord, airspeed, vario, trueAlt, alt, pos))

    def get_action_space(self) -> spaces.Box:

        return  spaces.Box(low= -1.0, high= 1.0, shape=(3, ), dtype=float)

    def get_default_action(self):

        return np.array([0, 0, 0], dtype='int')

    def send_control(self, control):

        self.env.control(control)

    def reset(self, seed=None, options=None):

        self.env.reset_env()

        obs = self.env.observation()
        
        return obs, {}

    def get_obs_rew_terminated_info(self):
        
        obs = self.env.observation()
        
        rew = self.env.reward(obs)

        term = self.env.end_episode(obs)
        
        return obs, rew, term, {}

    def wait(self):
        pass




MSFS_config = DEFAULT_CONFIG_DICT
MSFS_config["interface"] = MyRealTimeInterface
MSFS_config["time_step_duration"] = 0.2
MSFS_config["start_obs_capture"] = 1
MSFS_config["time_step_timeout_factor"] = 4.0
MSFS_config["ep_max_length"] = np.inf
MSFS_config["act_buf_len"] = 4
MSFS_config["reset_act_buf"] = False
MSFS_config["benchmark"] = True
MSFS_config["benchmark_polyak"] = 0.2
MSFS_config["disable_env_checking"] = True

terminated = False
env_interface = gym.make("real-time-gym-ts-v1", config = MSFS_config, )

ray.init(
    num_gpus= 1,
    include_dashboard=False,
    ignore_reinit_error=True,
    log_to_driver=False,
)

algorithm = (
    SAC(env= "real-time-gym-ts-v1", config= MSFS_config,)
    )

while not terminated:
    algorithm.train()

