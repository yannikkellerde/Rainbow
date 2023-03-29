import collections
import random
from math import sqrt

import numpy as np
import torch
from luxagent.Rainbow.common.utils import prep_observation_for_qnet


class PrioritizedReplayBuffer:
    """ based on https://nn.labml.ai/rl/dqn, supports n-step bootstrapping and parallel environments,
    removed alpha hyperparameter like google/dopamine
    """

    def __init__(self, burnin: int, capacity: int, gamma: float, device):
        self.burnin = burnin
        self.capacity = capacity  # must be a power of two
        self.gamma = gamma
        self.device = device


        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        self.max_priority = 1.0  # initial priority of new transitions

        self.data = [None for _ in range(self.capacity)]  # cyclical buffer for transitions
        self.next_idx = 0  # next write location
        self.size = 0  # number of buffer elements

    def prepare_transition(self,state_planes, state_features, next_state_planes, next_state_features, action: int, reward: float, done: bool):
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        return state_planes, state_features, next_state_planes, next_state_features, action, reward, done

    def put_whole_state(self, state:dict, action, reward, next_state:dict, done:bool, unit_dones:dict):
        for agent_id, player_state in state.items():
            for unit_id, unit_state in player_state.items():
                assert unit_id in reward[agent_id]
                if agent_id in next_state and unit_id in next_state[agent_id]:
                    s = unit_state
                    ns = next_state[agent_id][unit_id]
                    r = reward[agent_id][unit_id]
                    a = action[agent_id][unit_id]
                    d = done or (unit_id in unit_dones and unit_dones[unit_id])
                else:
                    s = unit_state
                    ns = unit_state
                    r = reward[agent_id][unit_id]
                    a = action[agent_id][unit_id]
                    d = True
                self.put(s["planes"],s["features"],ns["planes"],ns["features"],a,r,d)


    def put(self, state_planes, state_features, next_state_planes, next_state_features, action, reward, done):
        idx = self.next_idx
        self.data[idx] = self.prepare_transition(state_planes, state_features, next_state_planes, next_state_features, action, reward, float(done))
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

        self._set_priority_min(idx, sqrt(self.max_priority))
        self._set_priority_sum(idx, sqrt(self.max_priority))

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """ find the largest i such that the sum of the leaves from 1 to i is <= prefix sum"""

        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple:
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = indices[i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            weights[i] = weight / max_weight

        samples = []
        for i in indices:
            samples.append(self.data[i])

        return indices, weights, self.prepare_samples(samples)

    def prepare_samples(self, batch):
        state_planes, state_features, next_state_planes, next_state_features, action, reward, done = zip(*batch)

        state_planes, state_features, next_state_planes, next_state_features, action, reward, done = map(torch.stack, [state_planes, state_features, next_state_planes, next_state_features, action, reward, done])
        state_planes = state_planes.to(self.device)
        state_features = state_features.to(self.device)
        next_state_planes = next_state_planes.to(self.device)
        next_state_features = next_state_features.to(self.device)
        return state_planes,state_features,next_state_planes,next_state_features,action.squeeze(), reward.squeeze(), done.squeeze()

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = sqrt(priority)
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    @property
    def is_full(self):
        return self.capacity == self.size

    @property
    def burnedin(self):
        return len(self) >= self.burnin

    def __len__(self):
        return self.size

