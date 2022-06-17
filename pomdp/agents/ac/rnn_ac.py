import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pomdp.agents.ac.base import ActorCritic


class RnnCritic(nn.Module):
    def __init__(self, obs_dim):
        super(RnnCritic, self).__init__()
        self.linear1 = nn.Linear(obs_dim, 128)
        self.rnn = nn.RNN(128, 256, batch_first=True)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, obs, hidden):
        x = F.relu(self.linear1(obs))
        x, hidden = self.rnn(x, hidden)
        x = F.relu(x)
        x = self.linear3(x)
        return x, hidden


class RnnActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(RnnActor, self).__init__()
        self.act_dim = act_dim

        self.linear1 = nn.Linear(obs_dim, 128)
        self.rnn = nn.RNN(128, 256, batch_first=True)
        self.linear3 = nn.Linear(256, act_dim)

    def forward(self, obs, hidden):
        x = F.relu(self.linear1(obs))
        x, hidden = self.rnn(x, hidden)
        x = F.relu(x)
        x = self.linear3(x)
        dist = Categorical(F.softmax(x, dim=-1))
        return dist, hidden


class RnnActorCritic(ActorCritic):
    def __init__(
        self,
        env,
        gamma,
        seed,
        actor_lr,
        critic_lr,
        print_every,
        running_avg_rate,
        data_dir,
        h_dim,
    ):
        self.name = "RNN_AC" + f"_lr{actor_lr}"
        super(RnnActorCritic, self).__init__(
            env,
            gamma,
            seed,
            actor_lr,
            critic_lr,
            print_every,
            running_avg_rate,
            data_dir,
        )
        self.h_dim = h_dim
        self.ah = self.ch = None

    def get_actor(self):
        if self.env.finite_actions:
            return RnnActor(self.obs_dim, self.act_dim)

    def get_critic(self):
        return RnnCritic(self.obs_dim)

    def initialize_epoch(self, o):
        self.ah = torch.zeros((1, 1, self.h_dim)).to(self.device)
        self.ch = torch.zeros((1, 1, self.h_dim)).to(self.device)

    def step(self, o):
        o = torch.FloatTensor(o).unsqueeze(0).unsqueeze(0).to(self.device)
        dist, self.ah = self.actor(o, self.ah)
        value, self.ch = self.critic(o, self.ch)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return log_prob, a.item(), value
