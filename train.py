"""
This file includes the model and environment setup and the main training loop.
Look at the README.md file for details on how to use this.
"""

import matplotlib.pyplot as plt
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn

import torch, wandb
import numpy as np
from tqdm import trange
from rich import print
from luxagent.config import Rainbow_config, Env_config

from luxagent.Rainbow.common.rainbow import Rainbow
from luxagent.Rainbow.common.utils import LinearSchedule, StatisticsAccumlator, count_parameters
from luxagent.util.plotting import plot_robot_observation, get_planes_figure, get_planes_np_array
from dataclasses import asdict
from luxagent.env.mining_training_env import MiningEnv
import os
import random

torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm

def runRainbowDQN(rc:Rainbow_config,ec:Env_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = MiningEnv()
    state = env.reset()
    print('Done.')

    rainbow = Rainbow(env, rc, ec, device)
    # set up logging & model checkpoints
    wandb.init(project='luxai', save_code=True, config=dict(model_params=count_parameters(rainbow.q_policy),**asdict(rc),**asdict(ec)),
               mode=('online' if rc.use_wandb else 'offline'), anonymous='allow')
    if wandb.run.name is None:
        wandb.run.name = str(wandb.run.id)
    save_dir = os.path.join("checkpoints",wandb.run.name)
    os.makedirs(save_dir,exist_ok=True)

    # create decay schedules for dqn's exploration epsilon and per's importance sampling (beta) parameter
    eps_schedule = LinearSchedule(0, initial_value=rc.init_eps, final_value=rc.final_eps, decay_time=rc.eps_decay_frames)
    egoism_schedule = LinearSchedule(0, initial_value=rc.init_egoism_factor, final_value=rc.final_egoism_factor, decay_time=rc.egoism_factor_decay)
    shaping_schedule = LinearSchedule(0, initial_value=rc.init_shaping_factor, final_value=rc.final_shaping_factor, decay_time=rc.shaping_factor_decay)
    per_beta = rc.prioritized_er_beta0

    # When using many (e.g. 64) environments in parallel, having all of them be correlated can be an issue.
    # To avoid this, we estimate the mean episode length for this environment and then take i*(mean ep length/parallel envs count)
    # random steps in the i'th environment.
    # print(f'Creating', rc.parallel_envs, 'and decorrelating environment instances. This may take up to a few minutes.. ', end='')
    # decorr_steps = None
    # if rc.decorr:
        # decorr_steps = 1000 // rc.parallel_envs
    wandb.watch(rainbow.q_policy)

    print('[blue bold]Running environment =', str(env),
          '[blue bold]\nwith action space   =', env.action_space,
          '[blue bold]\nobservation space   =', env.observation_space,
          '[blue bold]\nand config:', sn(**asdict(rc)))

    episode_count = 0
    losses = deque(maxlen=10)
    q_values = deque(maxlen=10)
    grad_norms = deque(maxlen=10)
    iter_times = deque(maxlen=10)
    additional_statistics = {}
    stat_accumulator = StatisticsAccumlator()

    q_values_all = []
    planes_arrays = []

    # main training loop:
    t = trange(0, rc.training_frames + 1, rc.parallel_envs)
    for game_frame in t:
        iter_start = time.time()
        eps = 0 if (episode_count+10)%300<10 else eps_schedule(game_frame)
        env.egoism_factor = egoism_schedule(game_frame)
        env.shaping_factor = shaping_schedule(game_frame)

        # reset the noisy-nets noise in the policy
        if rc.noisy_dqn:
            rainbow.reset_noise(rainbow.q_policy)

        # compute actions to take in all parallel envs, asynchronously start environment step
        action = rainbow.act(state, eps)

        if rainbow.buffer.burnedin:
            for train_iter in range(rc.train_count):
                if rc.noisy_dqn and train_iter > 0: rainbow.reset_noise(rainbow.q_policy)
                q, loss, grad_norm = rainbow.train(rc.batch_size, beta=per_beta)
                losses.append(loss)
                grad_norms.append(grad_norm)
                q_values.append(q)
                q_values_all.append((game_frame, q))

        # copy the Q-policy weights over to the Q-target net
        # (see also https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/launcher.py#L155)
        if game_frame % rc.sync_dqn_target_every == 0 and rainbow.buffer.burnedin:
            rainbow.sync_Q_target()

        # block until environments are ready, then collect transitions and add them to the replay buffer
        next_state, reward, done, info = env.step(action)
        if episode_count%rc.plot_every_n_episodes == 0:# and env.step_number==100:
            if len(next_state["player_0"])>0:
                planes_arrays.append(get_planes_np_array(next(iter(next_state["player_0"].values()))["planes"]))
                # additional_statistics["plots/unit_vision"] = get_planes_figure(next(iter(next_state["player_0"].values()))["planes"])

        rainbow.buffer.put_whole_state(state,action,reward,next_state,done)
        state = next_state

        if done:
            statistics = env.get_statistics()
            state = env.reset()
            statistics["training/replay_buffer_size"] = len(rainbow.buffer)
            statistics["training/eps"] = eps
            statistics["training/shaping_factor"] = env.shaping_factor
            statistics["training/egoism_factor"] = env.egoism_factor
            statistics["training/mean_loss"] = np.mean(losses)
            statistics["training/mean_q"] = np.mean(q_values)
            statistics["timing/fps"] = np.mean(iter_times)
            statistics["training/grad_norm"] = np.mean(grad_norms)
            statistics["episode_number"] = episode_count
            stat_accumulator.update(statistics)
                
            if episode_count%rc.log_every_n_episodes == 0:
                to_log = stat_accumulator.get()
                if len(planes_arrays)>0:
                    planes_array = np.stack(planes_arrays)
                    print(planes_array.shape,planes_arrays[0].shape)
                    planes_array = np.transpose(planes_array,(0,3,1,2))
                    to_log["plot/episode_video"] = wandb.Video(planes_array,fps=4)
                to_log.update(additional_statistics)
                wandb.log(to_log)
            episode_count += 1
            additional_statistics = {}
            planes_arrays = []

        if game_frame % (50_000-(50_000 % rc.parallel_envs)) == 0:
            print(f' [{game_frame:>8} frames, {episode_count:>5} episodes]')
            if device!="cpu":
                torch.cuda.empty_cache()

        # every 1M frames, save a model checkpoint to disk and wandb
        if game_frame % (500_000-(500_000 % rc.parallel_envs)) == 0 and game_frame > 0:
            rainbow.save(game_frame, save_dir=save_dir, args=asdict(rc), run_name=wandb.run.name, run_id=wandb.run.id, q_values_all=q_values_all)
            print(f'Model saved at {game_frame} frames.')

        iter_times.append(time.time() - iter_start)
        t.set_description(f' [{game_frame:>8} frames, {episode_count:>5} episodes]', refresh=False)

    env.close()
    wandb.finish()
