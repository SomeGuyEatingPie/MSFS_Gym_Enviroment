import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac import SACConfig
import numpy as np
from rtgym import DEFAULT_CONFIG_DICT
import pathlib

from msfs_rt_env import DefaultEnv
from premade_env import GliderInterface

import logging

# Suppress erroneous warnings from the pysimconnect library
logging.getLogger("root").setLevel(40)


def trainer(interface: DefaultEnv):
    MSFS_config = DEFAULT_CONFIG_DICT
    MSFS_config["interface"] = interface
    MSFS_config["time_step_duration"] = 0.2
    MSFS_config["start_obs_capture"] = 0.0
    MSFS_config["time_step_timeout_factor"] = 0.8
    MSFS_config["ep_max_length"] = np.inf
    MSFS_config["act_buf_len"] = 6
    MSFS_config["reset_act_buf"] = False
    MSFS_config["benchmark"] = True
    MSFS_config["benchmark_polyak"] = 0.2
    MSFS_config["disable_env_checking"] = True

    path = pathlib.Path(__file__).parent.resolve()
    path_to_checkpoint = f"{path}\checkpoints"
    env_name = "real-time-gym-ts-v1"

    try:
        file = open("checkpoint_dir.txt", "r")
        path = file.readline()
        file.close()
        algo = Algorithm.from_checkpoint(path)
        print("Algorithm restored from checpoint")
    except:
        print("Training new policy")
        env = gym.make(env_name, config=MSFS_config)
        algo_config = (
            SACConfig()
            .resources(num_gpus=1)
            .environment(env=env_name, disable_env_checking=True)
        )
        algo = algo_config.build()

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


trainer(GliderInterface)
