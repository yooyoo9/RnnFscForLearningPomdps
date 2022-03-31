import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .base import ActorCritic


class LstmCritic(nn.Module):
    def __init__(self, obs_dim):
        super(LstmCritic, self).__init__()
        self.linear1 = nn.Linear(obs_dim, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, obs, hidden):
        x = F.relu(self.linear1(obs))
        x, hidden = self.lstm(x, hidden)
        x = F.relu(x)
        x = self.linear3(x)
        return x, hidden


class LstmActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(LstmActor, self).__init__()
        self.name = 'LSTM_AC'
        self.act_dim = act_dim

        self.linear1 = nn.Linear(obs_dim, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.linear3 = nn.Linear(256, act_dim)

    def forward(self, obs, hidden):
        x = F.relu(self.linear1(obs))
        x, hidden = self.lstm(x, hidden)
        x = F.relu(x)
        x = self.linear3(x)
        dist = Categorical(F.softmax(x, dim=-1))
        return dist, hidden


class LstmActorCritic(ActorCritic):
    def __init__(self, env, gamma, seed, actor_lr, critic_lr, print_every, running_avg_rate, data_dir, h_dim):
        self.name = 'LSTM_AC'
        super(LstmActorCritic, self).__init__(env, gamma, seed, actor_lr, critic_lr, print_every, running_avg_rate,
                                              data_dir)
        self.h_dim = h_dim
        self.a_hx = self.a_cx = self.c_hx = self.c_cx = None

    def get_actor(self):
        return LstmActor(self.obs_dim, self.act_dim)

    def get_critic(self):
        return LstmCritic(self.obs_dim)

    def initialize_epoch(self, o):
        self.a_hx = torch.zeros((1, 1, self.h_dim)).to(self.device)
        self.a_cx = torch.zeros((1, 1, self.h_dim)).to(self.device)
        self.c_hx = torch.zeros((1, 1, self.h_dim)).to(self.device)
        self.c_cx = torch.zeros((1, 1, self.h_dim)).to(self.device)

    def step(self, o):
        o = torch.FloatTensor(o).unsqueeze(0).unsqueeze(0).to(self.device)
        dist, (self.a_hx, self.a_cx) = self.actor(o, (self.a_hx, self.a_cx))
        value, (self.c_hx, self.c_cx) = self.critic(o, (self.c_hx, self.c_cx))
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return log_prob, a.item(), value

