"""
This files handles some of the internals for vectorized environments.
"""

import numpy as np
import gym

class vec_env(gym.Env):
    def __init__(self,env_creation_func,num_envs):
        super().__init__()
        self.envs = [env_creation_func() for _ in range(num_envs)]

