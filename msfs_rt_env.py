import gymnasium as gym
from gymnasium import spaces
from simconnect_env import MSFS
import ray 
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig
import numpy as np
from rtgym import RealTimeGymInterface, DEFAULT_CONFIG_DICT

import pathlib

import logging
logging.getLogger("root").setLevel(40)

class MyRealTimeInterface(RealTimeGymInterface):

    def __init__(self):
        
        self.env = MSFS()

    def get_observation_space(self) -> spaces.Tuple:

        heading = spaces.Box(low=-360.0, high=360.0, shape=(1,), dtype= np.float64)
        bank = spaces.Box(low=-360.0, high=360.0, shape=(1,), dtype= np.float64)
        pitch = spaces.Box(low=-360.0, high=360.0, shape=(1,), dtype= np.float64)
        turnCoord= spaces.Box(low=-127.0, high=127, shape=(1,), dtype= np.int64)
        airspeed = spaces.Box(low=-10.0, high=np.inf, shape=(1,), dtype= np.float64)
        vario = spaces.Box(low=-np.inf, high=200.0, shape=(1,), dtype= np.float64)
        trueAlt = spaces.Box(low=0.0, high=13000.0, shape=(1,), dtype= np.float64)
        alt = spaces.Box(low=0.0, high=13000.0, shape=(1,), dtype= np.float64)
        pos = spaces.Box(low=-180.0, high=180.0, shape=(2,), dtype= np.float64)

        return spaces.Tuple((heading, bank, pitch, turnCoord, airspeed, vario, trueAlt, alt, pos))

    def get_action_space(self) -> spaces.Box:

        return  spaces.Box(low= -1.0000, high= 1.0000, shape=(3, ), dtype= np.float64)

    def get_default_action(self):

        return np.array([0, 0, 0], dtype= np.float64)

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
MSFS_config["time_step_duration"] = 1
MSFS_config["start_obs_capture"] = 1
MSFS_config["time_step_timeout_factor"] = 1.2
MSFS_config["ep_max_length"] = np.inf
MSFS_config["act_buf_len"] = 4
MSFS_config["reset_act_buf"] = False
MSFS_config["benchmark"] = True
MSFS_config["benchmark_polyak"] = 0.2
MSFS_config["disable_env_checking"] = False


path = pathlib.Path(__file__).parent.resolve()
path_to_checkpoint = f"{path}\checkpoints"
env_name = "real-time-gym-ts-v1"

# env = gym.make(env_name, config = MSFS_config)
# algo_config = SACConfig().resources(num_gpus=1).environment(env= env_name, disable_env_checking=False)
# algo = algo_config.build()

try:
    file = open("checkpoint_dir.txt", "r")
    path = file.readline()
    file.close()
    algo = Algorithm.from_checkpoint(path)
    print("Algorithm restored from checpoint")
except:
    print("Training new policy")
    env = gym.make(env_name, config = MSFS_config)
    algo_config = SACConfig().resources(num_gpus=1).environment(env= env_name, disable_env_checking=False)
    algo = algo_config.build()

episode_reward = 0
terminated = truncated = False

while not terminated and not truncated:
    try:
        algo.train()
    finally:
        save_dir = algo.save(path_to_checkpoint)
        print(
            "An Algorithm checkpoint has been created inside directory: "
            f"'{save_dir}'."
        )
        file = open("checkpoint_dir.txt", "w")
        file.writelines(save_dir)
        file.close

algo.stop()


