"""
This file includes the model and environment setup and the main training loop.
Look at the README.md file for details on how to use this.
"""

import graph_tool.all    # Because import order matters apparently and not doing this results in error on some systems
from argparse import Namespace

from copy import deepcopy
import time, random
from collections import deque
from pathlib import Path
from types import SimpleNamespace as sn

import torch, wandb
import numpy as np

from rich import print

from common import argp
from common.rainbow import Rainbow
from common.utils import LinearSchedule
from graph_game.multi_env_manager import Env_manager, Debugging_manager
from alive_progress import alive_bar,alive_it
from GN0.visualize_transitions import visualize_transitions
from GN0.models import get_pre_defined
from common.utils import get_highest_model_path
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")
# torch.backends.cudnn.benchmark = True  # let cudnn heuristics choose fastest conv algorithm

if __name__ == '__main__':
    args, wandb_log_config = argp.read_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set up logging & model checkpoints
    wandb.init(project='rainbow_hex', save_code=True, config=dict(**wandb_log_config, log_version=100),
               mode=('online' if args.use_wandb else 'offline'), anonymous='allow', tags=args.wandb_tag.split(",") if args.wandb_tag else [])
    if args.use_wandb:
        save_dir = Path("checkpoints") / wandb.run.name
        save_dir.mkdir(parents=True)
        args.save_dir = str(save_dir)
    else:
        save_dir = Path("checkpoints") / "offline"
        save_dir.mkdir(parents=True, exist_ok=True)
        args.save_dir = str(save_dir)

    # create decay schedules for dqn's exploration epsilon and per's importance sampling (beta) parameter
    eps_schedule = LinearSchedule(0, initial_value=args.init_eps, final_value=args.final_eps, decay_time=args.eps_decay_frames)
    if args.prioritized_er_time == 0:
        per_beta_schedule = LinearSchedule(0, initial_value=args.prioritized_er_beta0, final_value=args.prioritized_er_beta0, decay_time=1)
    else:
        per_beta_schedule = LinearSchedule(0, initial_value=args.prioritized_er_beta0, final_value=1.0, decay_time=args.prioritized_er_time)


    # When using many (e.g. 64) environments in parallel, having all of them be correlated can be an issue.
    # To avoid this, we estimate the mean episode length for this environment and then take i*(mean ep length/parallel envs count)
    # random steps in the i'th environment.
    print(f'Creating', args.parallel_envs, 'and decorrelating environment instances.', end='')
    decorr_steps = 20
    # env_manager = Env_manager(args.parallel_envs,args.hex_size,gamma=args.gamma)
    env_manager = Env_manager(args.parallel_envs,args.hex_size,gamma=args.gamma,n_steps=[args.n_step],prune_exploratories=args.prune_exploratories)
    for _ in range(decorr_steps):
        env_manager.step(env_manager.sample())
    states = env_manager.observe()
    print('Done.')

    rainbow = Rainbow(env_manager, lambda :get_pre_defined(args.model_name,args), args)
    # args_cp = Namespace(**vars(args))
    # args_cp.noisy_dqn = True
    # old_model = get_pre_defined(args.model_name,args_cp).to(device)
    # elo = rainbow.join_elo_league("jumping-terrain",get_highest_model_path("jumping-terrain-85"),model=old_model)
    if args.load_model is not None:
        print("Loading model",args.load_model)
        stuff = torch.load(args.load_model)
        rainbow.q_policy.load_state_dict(stuff["state_dict"])
        if hasattr(rainbow.q_policy,"supports_cache") and rainbow.q_policy.supports_cache:
            rainbow.q_policy.import_norm_cache(*stuff["cache"])
        rainbow.opt.load_state_dict(stuff["optimizer_state_dict"])
        starting_frame = stuff["game_frame"]
    else:
        starting_frame = 0
    # wandb.watch(rainbow.maker_q_policy)
    # wandb.watch(rainbow.breaker_q_policy)

    print('[blue bold]\nconfig:', sn(**wandb_log_config))

    episode_count = 0
    stat_dict = {
            "targets":deque(maxlen=10),
            "losses":deque(maxlen=10),
            "q_values":deque(maxlen=10),
            "grad_norms":deque(maxlen=10),
            "training_rewards":deque(maxlen=10),
            "actions":deque(maxlen=10)
            }
    stats = {
            "maker":deepcopy(stat_dict),
            "breaker":deepcopy(stat_dict),
            "returns":deque(maxlen=100),
            "lengths":deque(maxlen=100)
            }

    growth_schedule = {6:(7,2,3600*3),7:(8,2,3600*5),8:(9,2,3600*7),9:(10,2,3600*9),10:(11,2,3600*11),11:(12,2,3600*1000)}

    returns_all = []
    q_values_all = []
    starting_states = states
    state_history,reward_history,done_history,action_history,exploratories_history = [],[],[],[],[]
    last_checkpoint = None
    hex_size = args.hex_size
    batch_size = args.batch_size + 64*(11-hex_size)

    checkpoint_frames = 160_000
    eval_frames = 80_000
    log_frames = 2_000
    jumping_extra = int(3600*1.7)

    if args.testing_mode:
        checkpoint_frames = 20_000
        eval_frames = 50_000
        jumping_extra = 10

    next_jump = time.perf_counter()+jumping_extra

    # main training loop:
    # we will do a total of args.training_frames/args.parallel_envs iterations
    # in each iteration we perform one interaction step in each of the args.parallel_envs environments,
    # and args.train_count training steps on batches of size args.batch_size
    with alive_bar(total=(args.training_frames+1)//args.parallel_envs, disable=False) as bar:
        for game_frame in range(starting_frame, args.training_frames+1, args.parallel_envs):
            additional_logs = {}
            iter_start = time.time()
            eps = eps_schedule(game_frame)
            per_beta = per_beta_schedule(game_frame)

            # reset the noisy-nets noise in the policy
            if args.noisy_dqn:
                rainbow.reset_noise(rainbow.q_policy)

            # compute actions to take in all parallel envs, asynchronously start environment step
            actions,exploratories = rainbow.act(states, eps)

            if env_manager.global_onturn == "m":
                stats["maker"]["actions"].append(torch.mean(actions.float()).item())
            else:
                stats["breaker"]["actions"].append(torch.mean(actions.float()).item())

            # if training has started, perform args.train_count training steps, each on a batch of size args.batch_size
            # print(rainbow.maker_buffer.burnedin,rainbow.breaker_buffer.burnedin)
            if rainbow.maker_buffer.burnedin and rainbow.breaker_buffer.burnedin:
                # print("Training",args.train_count)
                for train_iter in range(args.train_count):
                    if args.noisy_dqn and train_iter > 0: rainbow.reset_noise(rainbow.q_policy)
                    if (game_frame//args.parallel_envs)%2==0:  # Order matters, otherwise later one has advantage
                        order = ("breaker","maker")
                    else:
                        order = ("maker","breaker")
                    for player in order:
                        if player == "maker" and np.mean(stats["returns"])>0 and random.random()<np.mean(stats["returns"]):
                            # print("skipping maker training",np.mean(stats["returns"]))
                            continue
                        if player == "breaker" and np.mean(stats["returns"])<0 and random.random()<-np.mean(stats["returns"]):
                            # print("skipping breaker training",np.mean(stats["returns"]))
                            continue
                        q, targets, loss, grad_norm, reward = rainbow.train(batch_size, maker=player=="maker", beta=per_beta, add_cache=train_iter==args.train_count-1)
                        stats[player]["targets"].append(targets)
                        stats[player]["losses"].append(loss)
                        stats[player]["q_values"].append(q)
                        stats[player]["grad_norms"].append(grad_norm)
                        stats[player]["training_rewards"].append(reward)
                        q_values_all.append((game_frame, q))

            # copy the Q-policy weights over to the Q-target net
            # (see also https://github.com/spragunr/deep_q_rl/blob/master/deep_q_rl/launcher.py#L155)
            if game_frame % args.sync_dqn_target_every == 0 and rainbow.maker_buffer.burnedin and rainbow.breaker_buffer.burnedin:
                rainbow.sync_Q_target()

            # print("collect transitions")
            # block until environments are ready, then collect transitions and add them to the replay buffer
            # valid_actions = Env_manager.validate_actions(states,actions)
            valid_actions = Env_manager.validate_actions(states,actions)
            
            next_states, rewards, dones, infos = env_manager.step(valid_actions)
            # print("stepped")
            if len(state_history) >= args.num_required_repeated_actions:
                transitions_maker,transitions_breaker = env_manager.get_transitions(starting_states,state_history,action_history,reward_history,done_history,exploratories_history)
                for state, action, reward, next_state, done in transitions_breaker:
                    assert torch.all(state.x[:,2]==0) and torch.all(next_state.x[:,2]==0)
                    assert action<len(state.x)
                    rainbow.breaker_buffer.put_simple(state, action, reward, next_state, done)

                for state, action, reward, next_state, done in transitions_maker:
                    assert torch.all(state.x[:,2]==1) and torch.all(next_state.x[:,2]==1)
                    assert action<len(state.x)
                    rainbow.maker_buffer.put_simple(state, action, reward, next_state, done)

                starting_states = state_history[-(args.n_step+1)]
                state_history = state_history[-args.n_step:]
                action_history = action_history[-args.n_step:]
                reward_history = reward_history[-args.n_step:]
                done_history = done_history[-args.n_step:]
                exploratories_history = exploratories_history[-args.n_step:]
                next_states = [x.to(device) for x in next_states]

            state_history.append(next_states)
            reward_history.append(rewards)
            done_history.append(dones)
            action_history.append(actions)
            exploratories_history.append(exploratories)
            states = next_states
            # print("tryin to record metrics")

            # if any of the envs finished an episode, log stats to wandb
            for info, j in zip(infos, range(args.parallel_envs)):
                if 'episode_metrics' in info.keys():
                    episode_metrics = info['episode_metrics']
                    stats["returns"].append(episode_metrics['return'])
                    stats["lengths"].append(episode_metrics["length"])
                    returns_all.append((game_frame, episode_metrics['return']))

                    episode_count += 1


            if game_frame % (50_000-(50_000 % args.parallel_envs)) == 0:
                print(f' [{game_frame:>8} frames, {episode_count:>5} episodes] running average return = {np.mean(stats["returns"])}')
                torch.cuda.empty_cache()


            if game_frame % (checkpoint_frames-(checkpoint_frames % args.parallel_envs)) == 0:
                rainbow.disable_noise(rainbow.q_policy)
                last_checkpoint = rainbow.save(
                        game_frame, args=args, run_name=wandb.run.name, run_id=wandb.run.id,
                        target_metric=np.mean(stats["returns"]), returns_all=returns_all, q_values_all=q_values_all
                )
                elo = rainbow.join_elo_league(game_frame,last_checkpoint)
                additional_logs["elo"] = elo
                columns,data = rainbow.elo_handler.get_rating_table()
                additional_logs["rating_table"] = wandb.Table(columns = columns, data = data)
                plt.close("all")
                first_move_maker, first_move_breaker = rainbow.get_first_move_plots()
                additional_logs["first_move_maker"] = wandb.Image(first_move_maker)
                additional_logs["first_move_breaker"] = wandb.Image(first_move_breaker)
                print(f'Model saved at {game_frame} frames.')
                rainbow.reset_noise(rainbow.q_policy)

            if time.perf_counter()>next_jump and args.grow:
                if hex_size<11:
                    hex_size+=1
                    next_jump = time.perf_counter()+growth_schedule[hex_size][2]
                    batch_size = args.batch_size + 64*(11-hex_size)
                    print(f"Increasing hex size to {hex_size}")
                    rainbow.maximum_nodes = hex_size**2
                    rainbow.elo_handler.size = hex_size
                    env_manager.change_hex_size(hex_size)
                    starting_states = env_manager.observe()
                    rainbow.q_policy.grow_width(growth_schedule[hex_size][0]+rainbow.q_policy.gnn.hidden_channels)
                    rainbow.q_policy.grow_depth(growth_schedule[hex_size][1])
                    rainbow.q_target.grow_width(growth_schedule[hex_size][0]+rainbow.q_target.gnn.hidden_channels)
                    rainbow.q_target.grow_depth(growth_schedule[hex_size][1])
                    args.hidden_channels+=growth_schedule[hex_size][0]
                    args.num_layers+=growth_schedule[hex_size][1]
                    rainbow.elo_handler.elo_league_contestants = list()
                    last_checkpoint = None
                    model_creation_func = lambda :get_pre_defined(args.model_name,args)
                    rainbow.model_creation_func = model_creation_func
                    rainbow.elo_handler.create_empty_models(model_creation_func)
                    rainbow.opt = torch.optim.Adam(rainbow.q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
                    state_history,reward_history,done_history,action_history,exploratories_history = [],[],[],[],[]
                else:
                    next_jump = time.perf_counter()+100000000


            if game_frame % (eval_frames-(eval_frames % args.parallel_envs)) == 0 and rainbow.maker_buffer.burnedin:
                print("going for eval!")
                rainbow.disable_noise(rainbow.q_policy)
                if last_checkpoint is None:
                    additional_logs.update(rainbow.evaluate_models())
                else:
                    additional_logs.update(rainbow.evaluate_models(last_checkpoint))
                rainbow.reset_noise(rainbow.q_policy)
                print(additional_logs)

            if (game_frame % (log_frames-(log_frames % args.parallel_envs)) == 0 and game_frame > 0) or len(additional_logs)>0:
                log  = {}

                for key,value in stats.items():
                    if key in ("maker","breaker"):
                        for name,numbers in value.items():
                            log[key+"/"+name] = np.mean(numbers)
                    else:
                        log[key] = np.mean(value)
                log["buffer_size"] = len(rainbow.maker_buffer)
                log["hex_size"] = hex_size
                log["batch_size"] = batch_size
                log["game_frame"] = game_frame
                log["hidden_channels"] = args.hidden_channels
                log["num_layers"] = args.num_layers
                if eps > 0: log['epsilon'] = eps
                if len(additional_logs)>0:
                    for key,value in additional_logs.items():
                        log["ev/"+key] = value

                if args.prioritized_er: log['per_beta'] = per_beta
                wandb.log(log)

            bar.text = f' [{game_frame:>8} frames, {episode_count:>5} episodes]'
            bar()

    wandb.log({'x/game_frame': game_frame + args.parallel_envs, 'x/episode': episode_count,
               'x/train_step': (game_frame + args.parallel_envs) // args.parallel_envs * args.train_count,
               'x/emulator_frame': (game_frame + args.parallel_envs) * args.frame_skip})
    wandb.finish()
