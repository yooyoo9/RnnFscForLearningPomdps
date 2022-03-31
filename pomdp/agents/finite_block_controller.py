import numpy as np
import torch
from .base import ActorCritic, Actor, Critic


class FscActorCritic(ActorCritic):
    def __init__(self, env, gamma, seed, actor_lr, critic_lr, print_every, running_avg_rate, data_dir,
                 max_hist_len):
        self.name = 'FSC_AC' + str(max_hist_len)
        self.max_hist_len = max_hist_len
        self.o_buff = self.a_buff = None
        self.o_buff_len = 0
        super(FscActorCritic, self).__init__(env, gamma, seed, actor_lr, critic_lr, print_every, running_avg_rate,
                                             data_dir)

    def get_actor(self):
        in_dim = self.obs_dim + (self.obs_dim + 1) * self.max_hist_len
        return Actor(in_dim, self.act_dim)

    def get_critic(self):
        in_dim = (self.obs_dim + 1) * (self.max_hist_len + 1)
        return Critic(in_dim)

    def initialize_epoch(self, o):
        self.o_buff = np.zeros((self.max_hist_len, self.obs_dim))
        self.a_buff = np.zeros((self.max_hist_len, 1))
        self.o_buff[0, :] = o
        self.o_buff_len = 0

    def step(self, o):
        h_o = torch.FloatTensor(self.o_buff).view(1, self.o_buff.shape[0], self.o_buff.shape[1]).to(self.device)
        h_a = torch.FloatTensor(self.a_buff).view(1, self.a_buff.shape[0], self.a_buff.shape[1]).to(self.device)
        h = torch.cat([h_o, h_a], dim=-1).view(1, -1)
        o = torch.FloatTensor(o).unsqueeze(0).to(self.device)
        oh = torch.cat([o, h], dim=-1)
        dist = self.actor(oh)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        oah = torch.cat([o, a.unsqueeze(0), h], dim=-1)
        value = self.critic(oah)
        if self.o_buff_len == self.max_hist_len:
            self.o_buff[:self.max_hist_len - 1] = self.o_buff[1:]
            self.a_buff[:self.max_hist_len - 1] = self.a_buff[1:]
            self.o_buff[self.max_hist_len - 1] = o.cpu()
            self.a_buff[self.max_hist_len - 1] = a.cpu()
        else:
            self.o_buff[self.o_buff_len] = o.cpu()
            self.a_buff[self.o_buff_len] = a.cpu()
            self.o_buff_len += 1
        return log_prob, a.item(), value
