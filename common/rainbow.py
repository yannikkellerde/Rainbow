import random
from collections import defaultdict
from functools import partial
from typing import Tuple, Union

import numpy as np
import torch
import wandb
from torch import nn as nn
from rich import print

from luxagent.Rainbow.common import networks
from luxagent.Rainbow.common.replay_buffer import PrioritizedReplayBuffer

from luxagent.config import Rainbow_config, Env_config
from dataclasses import asdict
from luxagent.models.robot_Q import TwoBlockQ

class Rainbow:
    buffer: PrioritizedReplayBuffer

    def __init__(self, env, rc:Rainbow_config, ec:Env_config, device) -> None:
        self.device = device
        self.env = env
        self.max_action = ec.action_size-1

        self.q_policy = TwoBlockQ(**asdict(rc.q_config)).to(device)
        self.q_target = TwoBlockQ(**asdict(rc.q_config)).to(device)
        self.q_target.load_state_dict(self.q_policy.state_dict())

        #k = 0
        #for parameter in self.q_policy.parameters():
        #    k += parameter.numel()
        #print(f'Q-Network has {k} parameters.')

        self.double_dqn = rc.double_dqn

        self.prioritized_er = rc.prioritized_er
        if self.prioritized_er:
            self.buffer = PrioritizedReplayBuffer(rc.burnin, rc.buffer_size, rc.gamma,self.device)
        else:
            raise NotImplementedError()
            # self.buffer = UniformReplayBuffer(rc.burnin, rc.buffer_size, rc.gamma, rc.n_step, rc.parallel_envs, use_amp=self.use_amp)

        self.n_step_gamma = rc.gamma ** rc.n_step

        self.max_grad_norm = rc.max_grad_norm
        self.opt = torch.optim.Adam(self.q_policy.parameters(), lr=rc.lr)
        # self.scaler = GradScaler(enabled=self.use_amp)

        loss_fn_cls = nn.MSELoss if rc.loss_fn == 'mse' else nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(reduction=('none' if self.prioritized_er else 'mean'))

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    @torch.no_grad()
    def reset_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.disable_noise()

    def act(self, states:dict, eps:float):
        return self.act_multi_env([states],eps)[0]
    
    def act_multi_env(self, states:list, eps: float):
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        feature_list = []
        planes_list = []
        unit_ids_list = []
        agent_ids_list = []
        state_number_list = []
        for i,state in enumerate(states):
            for agent_id, state_player in state.items():
                for unit_id, robot_state in state_player.items():
                    planes_list.append(robot_state["planes"])
                    feature_list.append(robot_state["features"])
                    unit_ids_list.append(unit_id)
                    agent_ids_list.append(agent_id)
                    state_number_list.append(i)
        if len(feature_list) > 0: # Otherwise, no robots exist -> Do nothing
            features = torch.stack(feature_list).to(self.device)
            planes = torch.stack(planes_list).to(self.device)

            q_vec = self.q_policy(planes,features,advantages_only=True)
            best_actions = torch.argmax(q_vec,1).cpu()

        action = [defaultdict(dict) for _ in states]
        for i in range(len(agent_ids_list)):
            if random.random() < eps:
                action[state_number_list[i]][agent_ids_list[i]][unit_ids_list[i]] = random.randint(0,self.max_action)
            else:
                action[state_number_list[i]][agent_ids_list[i]][unit_ids_list[i]] = best_actions[i].item()
        return action

    @torch.no_grad()
    def td_target(self, reward: float, next_state_planes, next_state_features, done: bool):
        self.reset_noise(self.q_target)
        if self.double_dqn:
            best_action = torch.argmax(self.q_policy(next_state_planes, next_state_features, advantages_only=True), dim=1)
            next_Q = torch.gather(self.q_target(next_state_planes, next_state_features), dim=1, index=best_action.unsqueeze(1)).squeeze()
            return reward + self.n_step_gamma * next_Q * (1 - done)
        else:
            max_q = torch.max(self.q_target(next_state_planes,next_state_features), dim=1)[0]
            return reward + self.n_step_gamma * max_q * (1 - done)

    def train(self, batch_size, beta=None) -> Tuple[float, float, float]:
        if self.prioritized_er:
            indices, weights, (state_planes, state_features, next_state_planes, next_state_features, action, reward, done) = self.buffer.sample(batch_size, beta)
            weights = torch.from_numpy(weights).to(self.device)
        else:
            raise NotImplementedError()

        self.opt.zero_grad()
        td_est = torch.gather(self.q_policy(state_planes,state_features), dim=1, index=action.unsqueeze(1)).squeeze()
        td_tgt = self.td_target(reward, next_state_planes, next_state_features, done)

        if self.prioritized_er:
            td_errors = td_est-td_tgt
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6  # 1e-6 is the epsilon in PER
            self.buffer.update_priorities(indices, new_priorities)

            losses = self.loss_fn(td_tgt, td_est)
            loss = torch.mean(weights * losses)
        else:
            loss = self.loss_fn(td_tgt, td_est)

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(list(self.q_policy.parameters()), self.max_grad_norm)
        self.opt.step()

        return td_est.mean().item(), loss.item(), grad_norm.item()

    def save(self, game_frame, save_dir, **kwargs):
        save_path = (save_dir + f"/checkpoint_{game_frame}.pt")
        torch.save({**kwargs, 'state_dict': self.q_policy.state_dict(), 'game_frame': game_frame}, save_path)

        try:
            artifact = wandb.Artifact('saved_model', type='model')
            artifact.add_file(save_path)
            wandb.run.log_artifact(artifact)
            print(f'Saved model checkpoint at {game_frame} frames.')
        except Exception as e:
            print('[bold red] Error while saving artifacts to wandb:', e)
