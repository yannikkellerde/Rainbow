import random
import os
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

from common.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer

import math
from GN0.models import get_pre_defined,FactorizedNoisyLinear
from graph_game.multi_env_manager import Env_manager
from torch_geometric.data import Batch
from torch_scatter import scatter, scatter_mean, scatter_max
from GN0.RainbowDQN.evaluate_elo import Elo_handler, random_player
from graph_game.graph_tools_games import Hex_game
from GN0.util.convert_graph import convert_node_switching_game
from GN0.util.util import downsample_cnn_outputs
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: cuda not avaliabe, using cpu")

class Rainbow:
    maker_buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]
    breaker_buffer: Union[UniformReplayBuffer, PrioritizedReplayBuffer]

    def __init__(self, env:Env_manager, model_creation_func, args: SimpleNamespace) -> None:
        self.env:Env_manager = env
        self.cnn_mode = args.cnn_mode
        self.cnn_hex_size = args.cnn_hex_size
        if self.cnn_mode:
            self.starting_red_blue_sum = torch.sum(env.starting_obs[:2,:])
        self.save_dir = args.save_dir
        self.use_amp = args.use_amp
        self.maximum_nodes = args.hex_size**2
        self.hex_size = args.hex_size

        self.model_creation_func = model_creation_func
        self.q_policy = model_creation_func().to(device)
        self.q_target = model_creation_func().to(device)
        self.q_target.load_state_dict(self.q_policy.state_dict())

        self.elo_handler = Elo_handler(args.hex_size,empty_model_func=lambda :model_creation_func().to(device),device=device)
        self.elo_handler.add_player("maker",self.q_policy,set_rating=None,rating_fixed=True,can_join_roundrobin=False,uses_empty_model=True, cnn=self.cnn_mode, cnn_hex_size=self.cnn_hex_size if self.cnn_mode else None)
        self.elo_handler.add_player("breaker",self.q_policy,set_rating=None,rating_fixed=True,can_join_roundrobin=False,uses_empty_model=True, cnn=self.cnn_mode, cnn_hex_size=self.cnn_hex_size if self.cnn_mode else None)
        self.elo_handler.add_player("random",random_player,set_rating=0,simple=True,rating_fixed=True,can_join_roundrobin=True,uses_empty_model=False, cnn=self.cnn_mode, cnn_hex_size=self.cnn_hex_size if self.cnn_mode else None)

        self.roundrobin_players = args.roundrobin_players
        self.roundrobin_games = args.roundrobin_games

        self.double_dqn = args.double_dqn

        self.prioritized_er = args.prioritized_er
        if self.prioritized_er:
            self.maker_buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp, cnn_mode=self.cnn_mode)
        else:
            self.maker_buffer = UniformReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)

        if self.prioritized_er:
            self.breaker_buffer = PrioritizedReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp, cnn_mode=self.cnn_mode)
        else:
            self.breaker_buffer = UniformReplayBuffer(args.burnin, args.buffer_size, args.gamma, args.n_step, args.parallel_envs, use_amp=self.use_amp)

        self.n_step_gamma = args.gamma ** args.n_step

        self.max_grad_norm = args.max_grad_norm
        self.opt = torch.optim.Adam(self.q_policy.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scaler = GradScaler(enabled=self.use_amp)

        self.decay_lr = args.lr_decay_steps is not None
        if self.decay_lr: self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, (args.lr_decay_steps*args.train_count)//args.parallel_envs, gamma=args.lr_decay_factor)

        loss_fn_cls = nn.MSELoss if args.loss_fn == 'mse' else nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(reduction=('none' if self.prioritized_er else 'mean'))

    def get_first_move_plots(self):
        plt.clf()
        self.q_policy.eval()
        new_game = Hex_game(int(math.sqrt(self.maximum_nodes)))
        if self.cnn_mode:
            to_pred_maker = new_game.board.to_input_planes(self.cnn_hex_size).unsqueeze(0).to(device)
            new_game.view.gp["m"] = not new_game.view.gp["m"]
            to_pred_breaker = new_game.board.to_input_planes(self.cnn_hex_size).unsqueeze(0).to(device)
            pred_maker = downsample_cnn_outputs(self.q_policy(to_pred_maker),self.hex_size).squeeze()
            pred_breaker = downsample_cnn_outputs(self.q_policy(to_pred_breaker),self.hex_size).squeeze()
            maker_vinds = {new_game.board.board_index_to_vertex_index[int(i)]:value for i,value in enumerate(pred_maker)}
            breaker_vinds = {new_game.board.board_index_to_vertex_index[int(i)]:value for i,value in enumerate(pred_breaker)}
        else:
            to_pred_maker = convert_node_switching_game(new_game.view,global_input_properties=[1],need_backmap=True,old_style=True).to(device)
            to_pred_breaker = convert_node_switching_game(new_game.view,global_input_properties=[0],need_backmap=True,old_style=True).to(device)
            pred_maker = self.q_policy(to_pred_maker.x,to_pred_maker.edge_index).squeeze()
            pred_breaker = self.q_policy(to_pred_breaker.x,to_pred_breaker.edge_index).squeeze()
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
        self.q_policy.train()
        return fig_maker,fig_breaker
        

    def run_roundrobin_with_new_agent(self,game_frame,checkpoint,model=None):
        name = str(game_frame)+"_"+str(math.sqrt(self.maximum_nodes))
        self.elo_handler.add_player(name=name,checkpoint=checkpoint,model=model,episode_number=game_frame,uses_empty_model=True,cnn=self.cnn_mode, cnn_hex_size=self.cnn_hex_size if self.cnn_mode else None)
        return self.elo_handler.roundrobin(self.roundrobin_players,self.roundrobin_games,[name,"random"])
 

    def evaluate_models(self,checkpoint=None):
        self.q_policy.eval()
        additional_logs = {}
        maker_random = self.elo_handler.play_some_games("maker","random",num_games=128,temperature=0)
        additional_logs["maker_random_winrate"] = maker_random["maker"]/(maker_random["maker"]+maker_random["random"])
        breaker_random = self.elo_handler.play_some_games("random","breaker",num_games=128,temperature=0)
        additional_logs["breaker_random_winrate"] = breaker_random["breaker"]/(breaker_random["breaker"]+breaker_random["random"])
        maker_breaker = self.elo_handler.play_some_games("maker","breaker",num_games=128,temperature=0,random_first_move=True)
        additional_logs["maker_breaker_winrate"] = maker_breaker["maker"]/(maker_breaker["maker"]+maker_breaker["breaker"])

        all_stats = [maker_random,breaker_random,maker_breaker]
        if checkpoint is not None:
            stuff = torch.load(checkpoint)
            nn = self.model_creation_func().to(device)
            nn.load_state_dict(stuff["state_dict"])
            if "cache" in stuff and stuff["cache"] is not None:
                nn.import_norm_cache(*stuff["cache"])
            else:
                print("Warning, no cache")
            nn.eval()
            self.elo_handler.add_player("last_breaker",nn,set_rating=None,rating_fixed=True,can_join_roundrobin=False,uses_empty_model=True,cnn=self.cnn_mode,cnn_hex_size=self.cnn_hex_size if self.cnn_mode else None)
            maker_last_breaker = self.elo_handler.play_some_games("maker","last_breaker",num_games=128,temperature=0,random_first_move=True)
            additional_logs["maker_last_breaker_winrate"] = maker_last_breaker["maker"]/(maker_last_breaker["maker"]+maker_last_breaker["last_breaker"])

            self.elo_handler.add_player("last_maker",nn,set_rating=None,rating_fixed=True,can_join_roundrobin=False,uses_empty_model=True,cnn=self.cnn_mode,cnn_hex_size=self.cnn_hex_size if self.cnn_mode else None)
            breaker_last_maker = self.elo_handler.play_some_games("last_maker","breaker",num_games=128,temperature=0,random_first_move=True)
            additional_logs["breaker_last_maker_winrate"] = breaker_last_maker["breaker"]/(breaker_last_maker["breaker"]+breaker_last_maker["last_maker"])
            all_stats.extend([maker_last_breaker,breaker_last_maker])

        # self.elo_handler.score_some_statistics(all_stats)
        self.q_policy.train()

        return additional_logs

    def sync_Q_target(self) -> None:
        self.q_target.load_state_dict(self.q_policy.state_dict())

    @torch.no_grad()
    def reset_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()
        if hasattr(net,"maker_head"):
            for m in net.maker_head.modules():
                if isinstance(m, FactorizedNoisyLinear):
                    m.reset_noise()
        if hasattr(net,"breaker_head"):
            for m in net.breaker_head.modules():
                if isinstance(m, FactorizedNoisyLinear):
                    m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net) -> None:
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.disable_noise()
        if hasattr(net,"maker_head"):
            for m in net.maker_head.modules():
                if isinstance(m, FactorizedNoisyLinear):
                    m.disable_noise()
        if hasattr(net,"breaker_head"):
            for m in net.breaker_head.modules():
                if isinstance(m, FactorizedNoisyLinear):
                    m.disable_noise()

    def act_cnn(self, states, eps:float) -> Tuple[Tensor,Tensor]:
        self.q_policy.eval()
        batch = torch.stack(states).to(device)
        exploratories = np.zeros(len(states)).astype(bool)
        with torch.no_grad():
            action_values = downsample_cnn_outputs(self.q_policy(batch,advantages_only=True),self.hex_size)
            mask = downsample_cnn_outputs(torch.logical_or(batch[:,0].reshape(batch.shape[0],-1).bool(),batch[:,1].reshape(batch.shape[0],-1).bool()),self.hex_size)
            action_values[mask] = -5 # Exclude occupied squares

            actions = torch.argmax(action_values,dim=1)
            if eps > 0:
                for i in range(actions.shape[0]):
                    if torch.sum(states[i][:2,:])==self.starting_red_blue_sum or random.random() < eps: # First move: Random!
                        mask = downsample_cnn_outputs(torch.logical_not(torch.logical_or(states[i][0].flatten(),states[i][1].flatten())).to("cpu"),self.hex_size)
                        legal_actions = torch.arange(0,len(mask))[mask]
                        actions[i] = legal_actions[random.randint(0,len(legal_actions)-1)]
                        exploratories[i] = True
        self.q_policy.train()
        return actions.cpu(), exploratories

    def act(self, states, eps: float) -> Tuple[Tensor,Tensor]:
        """ computes an epsilon-greedy step with respect to the current policy self.q_policy """
        self.q_policy.eval()
        batch = Batch.from_data_list(states).to(device)
        exploratories = np.zeros(len(batch.ptr)-1).astype(bool)
        assert torch.all(batch.x[:,2] == states[0].x[0,2])
        with torch.no_grad():
            action_values = self.q_policy(batch.x,batch.edge_index,graph_indices=batch.batch, advantages_only=True)
            action_values[batch.ptr[:-1]] = -5
            action_values[batch.ptr[:-1]+1] = -5
            _,actions = scatter_max(action_values,batch.batch,dim=0)
            actions = actions.squeeze()-batch.ptr[:-1]
            if 1 in actions or 0 in actions:
                evil = (torch.logical_or(actions==0,actions == 1).nonzero(as_tuple=True)[0])
                for e in evil:
                    print(action_values[batch.batch==e])
                    print(batch.x[batch.batch==e])
                raise ValueError("Selected evil vertex")

            if eps > 0:
                for i in range(actions.shape[0]):
                    if (batch.ptr[i+1]-batch.ptr[i]-2)==self.maximum_nodes or random.random() < eps: # First move: Random!
                        actions[i] = random.randint(2,int(torch.sum(batch.batch==i).item())-1)
                        exploratories[i] = True
            self.q_policy.train()
            return actions.cpu(), exploratories

    @torch.no_grad()
    def cnn_td_target(self, reward: Tensor, next_state:Tensor, done: Tensor):
        self.reset_noise(self.q_target)
        if self.double_dqn:
            advantages = downsample_cnn_outputs(self.q_policy(next_state,advantages_only=True),self.hex_size)
            best_action = torch.argmax(advantages,dim=1)
            best_action = best_action.squeeze()
            next_Q = torch.gather(downsample_cnn_outputs(self.q_target(next_state),self.hex_size),1,best_action.unsqueeze(1)).squeeze()
            return reward + self.n_step_gamma * next_Q * (1 - done)
        else:
            advantages = downsample_cnn_outputs(self.q_target(next_state),self.hex_size)
            max_q = torch.max(advantages,dim=1)
            return reward + self.n_step_gamma * max_q * (1 - done)

    @torch.no_grad()
    def td_target(self, reward: Tensor, next_state:Batch, done: Tensor):
        self.reset_noise(self.q_target)
        if self.double_dqn:
            advantages = self.q_policy(next_state.x,next_state.edge_index,graph_indices=next_state.batch,advantages_only=True)
            advantages[next_state.ptr[:-1]] = -5
            advantages[next_state.ptr[:-1]+1] = -5
            _,best_action = scatter_max(advantages,next_state.batch,dim=0)
            best_action = best_action.squeeze()
            next_Q = self.q_target(next_state.x,next_state.edge_index,graph_indices=next_state.batch,ptr=next_state.ptr).squeeze()[best_action]
            return reward + self.n_step_gamma * next_Q * (1 - done)
        else:
            advantages = self.q_target(next_state.x,next_state.edge_index,graph_indices=next_state.batch,ptr=next_state.ptr)
            advantages[next_state.ptr[:-1]] = -5
            advantages[next_state.ptr[:-1]+1] = -5
            max_q,_ = scatter_max(advantages,next_state.batch,dim=0)
            return reward + self.n_step_gamma * max_q * (1 - done)

    def train(self, batch_size, maker:bool, beta=None, add_cache=False) -> Tuple[float,float, float, float, float]:
        if maker:
            buffer = self.maker_buffer
        else:
            buffer = self.breaker_buffer
        self.q_policy.train()
        if self.prioritized_er:
            indices, weights, (state, next_state, action, reward, done) = buffer.sample(batch_size,beta=beta)
            weights = torch.from_numpy(weights).cuda()
            reward_mean = torch.mean((weights*reward)[reward!=0]).cpu().item()
        else:
            state, next_state, action, reward, done = buffer.sample(batch_size,beta=beta)
            reward_mean = torch.mean(reward[reward!=0]).cpu().item()

        if not self.cnn_mode:
            assert torch.all(state.x[:,2]==int(maker)) and torch.all(next_state.x[:,2]==int(maker))
        self.opt.zero_grad()
        with autocast(enabled=self.use_amp):
            if self.cnn_mode:
                p_res = downsample_cnn_outputs(self.q_policy(state),self.hex_size)
                td_est = torch.gather(p_res,1,action.unsqueeze(1)).squeeze()
                td_tgt = self.cnn_td_target(reward, next_state, done)

            else:
                true_action = action+state.ptr[:-1]
                p_res = self.q_policy(state.x,state.edge_index,graph_indices=state.batch,ptr=state.ptr,set_cache=add_cache)
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
        grad_norm = nn.utils.clip_grad_norm_(list(self.q_policy.parameters()), self.max_grad_norm).item()
        # parameters = [p for p in q_policy.parameters() if p.grad is not None and p.requires_grad]
        # grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2).item()
        # self.scaler.step(opt)
        # self.scaler.update()
        self.opt.step()
        self.opt.zero_grad()

        if self.decay_lr:
            self.scheduler.step()
        
        if self.prioritized_er and False:
            est_mean = (td_est*weights).mean().item()
            gt_mean = (td_tgt*weights).mean().item()
        else:
            est_mean = (td_est).mean().item()
            gt_mean = (td_tgt).mean().item()

        return est_mean, gt_mean, loss.item(), grad_norm, reward_mean

    def save_model(self, game_frame, hex_size, **kwargs):
        save_path = os.path.join(self.save_dir,str(hex_size),f"checkpoint_{game_frame}.pt")
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        stuff = {**kwargs, 'state_dict': self.q_policy.state_dict(), 'game_frame':game_frame, 'optimizer_state_dict':self.opt.state_dict()}
        if not self.cnn_mode and self.q_policy.gnn.has_cache:
            stuff['cache'] = self.q_policy.export_norm_cache()
        torch.save(stuff,save_path)

        return save_path

    def save(self, game_frame, hex_size, **kwargs):
        # save_path = (self.save_dir + f"/checkpoint_{game_frame}.pt")
        return self.save_model(game_frame,hex_size,**kwargs)

