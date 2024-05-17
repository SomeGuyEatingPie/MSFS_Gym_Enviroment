# MSFS Gym Environment  
Create custom [real-time](https://github.com/yannbouteiller/rtgym/tree/main) - [Gymnasium](https://gymnasium.farama.org) environments for Microsoft Flight Simulator 2020.   

MSFS Gym Environemnt is a simple framework for building RL environemnts using the efficient real-time threaded framework: [Real-Time Gym](https://github.com/yannbouteiller/rtgym) ```rtgym```.
It is coded in python.

## Requirements 
[Gymnasium 0.28.1](https://gymnasium.farama.org/index.html)  
[rtgym 0.13](https://github.com/yannbouteiller/rtgym/tree/main) – MSFS_Gym_Env is built on top of rtgym.  
[pysimconnect 0.2.6](https://github.com/patricksurry/pysimconnect) - Python wrapper for the SimConnect API.  

### Optional
[ray 2.4.0](https://www.ray.io/) - rllib used by the glider enviroment demo (msfs_soar_training.py)

## Installation
to install all requirements use:  
```bash
pip install -r requirements.txt
```


## Tutorial
Custom environemnts must inherit from [DefaultEnv](https://github.com/SomeGuyEatingPie/MSFS_Gym_Enviroment/blob/master/default_env.py) and follow the same class interface, exactly as shown below.
```python
class CustomEnv():

    def __init__(self):
        super().__init__()
        ...

    def get_observation_space(self) -> spaces.Tuple:
      ...
    def get_action_space(self) -> spaces.Box:
      ...
    def get_default_action(self) -> np.array:
      ...
    def observation(self):
      ...
    def control(self, controlInput):
      ...
    def reset_env(self, seed=None, options=None):
      ...
    def reward(self, observation):
      ...
    def terminate(self, observation):

```
Write your own environment code in this class, a working example of this can be seen in [premade_env.py](https://github.com/SomeGuyEatingPie/MSFS_Gym_Enviroment/blob/master/premade_env.py).  

Then copy an interface class for your custom environment, which inherits from [RealTimeFlightSimInterface](https://github.com/SomeGuyEatingPie/MSFS_Gym_Enviroment/blob/master/msfs_rt_env.py). 
```python
class CusatomInterface(RealTimeFlightSimInterface):

    def __init__(self):
        super().__init__(CustomEnv())
```
The interface for the previous example is also found in [premade_env.py](https://github.com/SomeGuyEatingPie/MSFS_Gym_Enviroment/blob/master/premade_env.py).  

To instantiate the environemnt, use:
```python
import gymnasium as gym
from rtgym import DEFAULT_CONFIG_DICT

custom_config = DEFAULT_CONFIG_DICT
custom_config["interface"] = CustomInterface
custom_config["..."] = ...
...

env = gym.make("real-time-gym-ts-v1", config = custom_config)
```



## Authors  
Contributions welcome.  
Please submit a pull request and add yourself to the Contributors list.  
  
### Maintainers  
- Sean Johnson  
  
### Contributors  
  
### Acknowledgements  
Yann Bouteiller - rtgym  
Patrick Surry - pysimconnect  
Dr Nonso Nnamoko (Edge Hill University) - Project Supervisor  
