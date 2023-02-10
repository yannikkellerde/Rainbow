import zlib
from copy import deepcopy
import torch
from tqdm.auto import trange
import os
from math import sqrt

def get_highest_model_path(tagname):
    cdir = os.path.dirname(os.path.abspath(__file__))
    stuff_dir = os.path.join(cdir,"..","checkpoints",tagname)
    checkpoints = os.listdir(stuff_dir)
    checkpoints.sort(key=lambda x:-int(x.split("_")[1].split(".")[0]))
    return os.path.join(stuff_dir,checkpoints[0])

def prep_observation_for_qnet(tensor, use_amp):
    """ Tranfer the tensor the gpu and reshape it into (batch, frame_stack*channels, y, x) """
    assert len(tensor.shape) == 5, tensor.shape # (batch, frame_stack, y, x, channels)
    tensor = tensor.cuda().permute(0, 1, 4, 2, 3) # (batch, frame_stack, channels, y, x)
    # .cuda() needs to be before this ^ so that the tensor is made contiguous on the gpu
    tensor = tensor.reshape((tensor.shape[0], tensor.shape[1]*tensor.shape[2], *tensor.shape[3:]))

    return tensor.to(dtype=(torch.float16 if use_amp else torch.float32)) / 255

class LinearSchedule:
    """Set up a linear hyperparameter schedule (e.g. for dqn's epsilon parameter)"""

    def __init__(self, burnin: int, initial_value: float, final_value: float, decay_time: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_time = decay_time
        self.burnin = burnin

    def __call__(self, frame: int) -> float:
        if frame < self.burnin:
            return self.initial_value
        else:
            frame = frame - self.burnin

        slope = (self.final_value - self.initial_value) / self.decay_time
        if self.final_value < self.initial_value:
            return max(slope * frame + self.initial_value, self.final_value)
        else:
            return min(slope * frame + self.initial_value, self.final_value)


def env_seeding(user_seed, env_name):
    return user_seed + zlib.adler32(bytes(env_name, encoding='utf-8')) % 10000
