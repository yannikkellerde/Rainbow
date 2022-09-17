import random
import matplotlib.pyplot as plt
from functools import partial
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np
import torch
import wandb
from torch import nn as nn
from rich import print
from torch.cuda.amp import GradScaler, autocast
from torch import LongTensor,Tensor

from common import networks
from common.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
from common.utils import prep_observation_for_qnet

import math
from GN0.models import Duelling,get_pre_defined
from GN0.util import visualize_graph
from graph_game.multi_env_manager import Env_manager
from torch_geometric.data import Batch
from torch_scatter import scatter, scatter_mean, scatter_max
from GN0.evaluate_elo import Elo_handler, random_player
from graph_game.graph_tools_games import Hex_game
from GN0.convert_graph import convert_node_switching_game
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

class Rainbow:
    maker_buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]
    breaker_buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]

    def __init__(self, env:Env_manager, model_creation_func, args: SimpleNamespace) -> None:
        self.env:Env_manager = env
        self.save_dir = args.save_dir
        self.use_amp = args.use_amp
        self.maximum_nodes = args.hex_size**2

        self.model_creation_func = model_creation_func
        self.maker_q_policy = model_creation_func().to(device)
        self.maker_q_target = model_creation_func().to(device)
        self.maker_q_target.load_state_dict(self.maker_q_policy.state_dict())

        self.breaker_q_policy = model_creation_func().to(device)
        self.breaker_q_target = model_creation_func().to(device)
        self.breaker_q_target.load_state_dict(self.breaker_q_policy.state_dict())

        self.elo_handler = Elo_handler(args.hex_size,empty_model_func=lambda :model_creation_func().to(device),device=device)
        self.elo_handler.add_player("maker",self.maker_q_policy,fix_rating=1500)
        self.elo_handler.add_player("breaker",self.breaker_q_policy,fix_rating=1500)
        self.elo_handler.add_player("random",random_player,fix_rating=1500,simple=True)

        self.double_dqn = args.double_dqn

        self.prioritized_er = args.prioritized_er
        if self.prioritized_er:
            self.maker_buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)
        else:
            self.maker_buffer = UniformReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)

        if self.prioritized_er:
            self.breaker_buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)
        else:
            self.breaker_buffer = UniformReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)

        self.n_step_gamma = args.gamma ** args.n_step

        self.max_grad_norm = args.max_grad_norm
        self.breaker_opt = torch.optim.Adam(self.breaker_q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.maker_opt = torch.optim.Adam(self.maker_q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.decay_lr = args.lr_decay_steps is not None
        if self.decay_lr: self.breaker_scheduler = torch.optim.lr_scheduler.StepLR(self.breaker_opt, (args.lr_decay_steps*args.train_count)//args.parallel_envs, gamma=args.lr_decay_factor)
        if self.decay_lr: self.maker_scheduler = torch.optim.lr_scheduler.StepLR(self.maker_opt, (args.lr_decay_steps*args.train_count)//args.parallel_envs, gamma=args.lr_decay_factor)

        loss_fn_cls = nn.MSELoss if args.loss_fn == 'mse' else nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(reduction=('none' if self.prioritized_er else 'mean'))

    def get_first_move_plots(self):
        plt.clf()
        self.maker_q_policy.eval()
        self.breaker_q_policy.eval()
        new_game = Hex_game(int(math.sqrt(self.maximum_nodes)))
        to_pred_maker = convert_node_switching_game(new_game.view,global_input_properties=[1],need_backmap=True).to(device)
        to_pred_breaker = convert_node_switching_game(new_game.view,global_input_properties=[0],need_backmap=True).to(device)
        pred_maker = self.maker_q_policy(to_pred_maker.x,to_pred_maker.edge_index).squeeze()
        pred_breaker = self.breaker_q_policy(to_pred_breaker.x,to_pred_breaker.edge_index).squeeze()
        print(pred_maker,pred_breaker)
        maker_vinds = {to_pred_maker.backmap[int(i)]:value for i,value in enumerate(pred_maker) if int(i)>1}
        breaker_vinds = {to_pred_breaker.backmap[int(i)]:value for i,value in enumerate(pred_breaker) if int(i)>1}
        maker_vprop = new_game.view.new_vertex_property("float")
        breaker_vprop = new_game.view.new_vertex_property("float")
        for key,value in maker_vinds.items():
            maker_vprop[new_game.view.vertex(key)] = value

        for key,value in breaker_vinds.items():
            breaker_vprop[new_game.view.vertex(key)] = value

        fig_maker = new_game.board.matplotlib_me(vprop=maker_vprop,color_based_on_vprop=True)
        fig_breaker = new_game.board.matplotlib_me(vprop=breaker_vprop,color_based_on_vprop=True)

        self.maker_q_policy.train()
        self.breaker_q_policy.train()
        return fig_maker,fig_breaker
        

    def join_elo_league(self,game_frame,maker_checkpoint,breaker_checkpoint):
        return self.elo_handler.add_elo_league_contestant(str(game_frame)+"_"+str(math.sqrt(self.maximum_nodes)),maker_checkpoint,breaker_checkpoint)
 

    def evaluate_models(self,last_maker_checkpoint=None,last_breaker_checkpoint=None):
        self.maker_q_policy.eval()
        self.breaker_q_policy.eval()
        additional_logs = {}
        maker_random = self.elo_handler.play_some_games("maker","random",num_games=128,temperature=0)
        additional_logs["maker_random_winrate"] = maker_random["maker"]/(maker_random["maker"]+maker_random["random"])
        breaker_random = self.elo_handler.play_some_games("random","breaker",num_games=128,temperature=0)
        additional_logs["breaker_random_winrate"] = breaker_random["breaker"]/(breaker_random["breaker"]+breaker_random["random"])
        maker_breaker = self.elo_handler.play_some_games("maker","breaker",num_games=128,temperature=0,random_first_move=True)
        additional_logs["maker_breaker_winrate"] = maker_breaker["maker"]/(maker_breaker["maker"]+maker_breaker["breaker"])

        if last_breaker_checkpoint is not None:
            stuff = torch.load(last_breaker_checkpoint)
            nn = self.model_creation_func().to(device)
            nn.load_state_dict(stuff["state_dict"])
            if "cache" in stuff and stuff["cache"] is not None:
                nn.import_norm_cache(*stuff["cache"])
            else:
                print("Warning, no cache")
            nn.eval()
            self.elo_handler.add_player("last_breaker",nn,fix_rating=1500)
            maker_last_breaker = self.elo_handler.play_some_games("maker","last_breaker",num_games=128,temperature=0,random_first_move=True)
            additional_logs["maker_last_breaker_winrate"] = maker_last_breaker["maker"]/(maker_last_breaker["maker"]+maker_last_breaker["last_breaker"])

        if last_maker_checkpoint is not None:
            stuff = torch.load(last_maker_checkpoint)
            nn = self.model_creation_func().to(device)
            nn.load_state_dict(stuff["state_dict"])
            if "cache" in stuff and stuff["cache"] is not None:
                nn.import_norm_cache(*stuff["cache"])
            else:
                print("Warning, no cache")
            nn.eval()
            self.elo_handler.add_player("last_maker",nn,fix_rating=1500)
            breaker_last_maker = self.elo_handler.play_some_games("last_maker","breaker",num_games=128,temperature=0,random_first_move=True)
            additional_logs["breaker_last_maker_winrate"] = breaker_last_maker["breaker"]/(breaker_last_maker["breaker"]+breaker_last_maker["last_maker"])

        self.maker_q_policy.train()
        self.breaker_q_policy.train()

        return additional_logs

    def sync_Q_target(self) -> None:
        self.maker_q_target.load_state_dict(self.maker_q_policy.state_dict())
        self.breaker_q_target.load_state_dict(self.breaker_q_policy.state_dict())

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

    def act(self, states, eps: float) -> Tensor:
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        if states[0].x[0,2] == 1:
            q_policy = self.maker_q_policy
        else:
            q_policy = self.breaker_q_policy
        q_policy.eval()
        batch = Batch.from_data_list(states).to(device)
        assert torch.all(batch.x[:,2] == states[0].x[0,2])
        with torch.no_grad():
            action_values = q_policy(batch.x,batch.edge_index,graph_indices=batch.batch, advantages_only=True)
            action_values[batch.ptr[:-1]] = -3
            action_values[batch.ptr[:-1]+1] = -3
            _,actions = scatter_max(action_values,batch.batch,dim=0)
            actions = actions.squeeze()-batch.ptr[:-1]
            if 1 in actions or 0 in actions:
                evil = (torch.logical_or(actions==0,actions == 1).nonzero(as_tuple=True)[0])
                print(action_values[batch.batch==evil])
                raise ValueError("Selected evil vertex")

            if eps > 0:
                for i in range(actions.shape[0]):
                    if (batch.ptr[i+1]-batch.ptr[i]-2)==self.maximum_nodes or random.random() < eps: # First move: Random!
                        actions[i] = random.randint(2,int(torch.sum(batch.batch==i).item())-1)
            q_policy.train()
            return actions.cpu()

    @torch.no_grad()
    def td_target(self, reward: Tensor, next_state:Batch, done: Tensor):
        if next_state.x[0,2] == 1:
            q_policy = self.maker_q_policy
            q_target = self.maker_q_target
        else:
            q_policy = self.breaker_q_policy
            q_target = self.breaker_q_target
        self.reset_noise(q_target)
        if self.double_dqn:
            advantages = q_policy(next_state.x,next_state.edge_index,graph_indices=next_state.batch,advantages_only=True)
            advantages[next_state.ptr[:-1]] = -3
            advantages[next_state.ptr[:-1]+1] = -3
            _,best_action = scatter_max(advantages,next_state.batch,dim=0)
            best_action = best_action.squeeze()
            next_Q = q_target(next_state.x,next_state.edge_index,graph_indices=next_state.batch,ptr=next_state.ptr).squeeze()[best_action]
            return reward + self.n_step_gamma * next_Q * (1 - done)
        else:
            advantages = q_target(next_state.x,next_state.edge_index,graph_indices=next_state.batch,ptr=next_state.ptr)
            advantages[next_state.ptr[:-1]] = -3
            advantages[next_state.ptr[:-1]+1] = -3
            max_q,_ = scatter_max(advantages,next_state.batch,dim=0)
            return reward + self.n_step_gamma * max_q * (1 - done)

    def train(self, batch_size, maker:bool, beta=None, add_cache=False) -> Tuple[float,float, float, float, float]:
        if maker:
            q_policy = self.maker_q_policy
            buffer = self.maker_buffer
            opt = self.maker_opt
        else:
            q_policy = self.breaker_q_policy
            buffer = self.breaker_buffer
            opt = self.breaker_opt
        q_policy.train()
        if self.prioritized_er:
            indices, weights, (state, next_state, action, reward, done) = buffer.sample(batch_size,beta=beta)
            weights = torch.from_numpy(weights).cuda()
            reward_mean = torch.mean((weights*reward)[reward!=0]).cpu().item()
        else:
            state, next_state, action, reward, done = buffer.sample(batch_size,beta=beta)
            reward_mean = torch.mean(reward[reward!=0]).cpu().item()

        assert torch.all(state.x[:,2]==int(maker)) and torch.all(next_state.x[:,2]==int(maker))
        opt.zero_grad()
        with autocast(enabled=self.use_amp):
            true_action = action+state.ptr[:-1]
            p_res = q_policy(state.x,state.edge_index,graph_indices=state.batch,ptr=state.ptr,set_cache=add_cache)
            td_est = p_res[true_action]
            td_tgt = self.td_target(reward, next_state, done)

            if self.prioritized_er:
                td_errors = td_est-td_tgt
                new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6  # 1e-6 is the epsilon in PER
                buffer.update_priorities(indices, new_priorities)

                losses = self.loss_fn(td_tgt, td_est)
                loss = torch.mean(weights * losses)
            else:
                loss = self.loss_fn(td_tgt, td_est)

        loss.backward()
        # self.scaler.scale(loss).backward()

        # self.scaler.unscale_(opt)
        grad_norm = nn.utils.clip_grad_norm_(list(q_policy.parameters()), self.max_grad_norm).item()
        # parameters = [p for p in q_policy.parameters() if p.grad is not None and p.requires_grad]
        # grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2).item()
        # self.scaler.step(opt)
        # self.scaler.update()
        opt.step()
        opt.zero_grad()

        if self.decay_lr:
            if maker:
                self.maker_scheduler.step()
            else:
                self.breaker_scheduler.step()
        
        if self.prioritized_er and False:
            est_mean = (td_est*weights).mean().item()
            gt_mean = (td_tgt*weights).mean().item()
        else:
            est_mean = (td_est).mean().item()
            gt_mean = (td_tgt).mean().item()

        return est_mean, gt_mean, loss.item(), grad_norm, reward_mean

    def save_model(self, game_frame, **kwargs):
        save_path_maker = (self.save_dir + f"/checkpoint_maker_{game_frame}.pt")
        stuff = {**kwargs, 'state_dict': self.maker_q_policy.state_dict(), 'game_frame':game_frame, 'optimizer_state_dict':self.maker_opt.state_dict()}
        if self.maker_q_policy.gnn.has_cache:
            stuff['cache'] = self.maker_q_policy.export_norm_cache()
        torch.save(stuff,save_path_maker)

        save_path_breaker = (self.save_dir + f"/checkpoint_breaker_{game_frame}.pt")
        stuff = {**kwargs, 'state_dict': self.breaker_q_policy.state_dict(), 'game_frame':game_frame, 'optimizer_state_dict':self.breaker_opt.state_dict()}
        if self.breaker_q_policy.gnn.has_cache:
            stuff['cache'] = self.breaker_q_policy.export_norm_cache()
        torch.save(stuff,save_path_breaker)
        return save_path_maker,save_path_breaker

    def save(self, game_frame, **kwargs):
        # save_path = (self.save_dir + f"/checkpoint_{game_frame}.pt")
        return self.save_model(game_frame,**kwargs)

        # try:
        #     artifact = wandb.Artifact('saved_model', type='model')
        #     artifact.add_file(save_path)
        #     wandb.run.log_artifact(artifact)
        #     print(f'Saved model checkpoint at {game_frame} frames.')
        # except Exception as e:
        #     print('[bold red] Error while saving artifacts to wandb:', e)
