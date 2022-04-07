import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Categorical
from pomdp.utils.logger import EpochLogger, setup_logger_kwargs


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(obs_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.act_dim = act_dim

        self.linear1 = nn.Linear(obs_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        dist = Categorical(F.softmax(x, dim=-1))
        return dist


class ActorCritic:
    def __init__(self, env, gamma, seed, actor_lr, critic_lr, print_every, running_avg_rate, data_dir):
        if not hasattr(self, 'name'):
            self.name = 'AC'
        logger_kwargs = setup_logger_kwargs(self.name, env.name, seed, data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.print_every = print_every
        self.running_avg_rate = running_avg_rate

        self.obs_dim = env.observation_space_n
        self.act_dim = env.action_space_n
        self.max_ep_len = env.max_ep_len

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = self.get_actor().to(self.device)
        self.critic = self.get_critic().to(self.device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

    def get_actor(self):
        return Actor(self.obs_dim, self.act_dim)

    def get_critic(self):
        return Critic(self.obs_dim)

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            returns.insert(0, R)
        return returns

    def step(self, o):
        o = torch.FloatTensor(o).unsqueeze(0).unsqueeze(0).to(self.device)
        dist = self.actor(o)
        value = self.critic(o)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return log_prob, a.item(), value

    def initialize_epoch(self, o):
        return

    def train(self, epochs):
        start_time = time.time()
        running_reward = 0.0
        ep_rewards = []
        for ep in range(epochs):
            o, ep_ret, ep_len = self.env.reset(seed=self.seed), 0, 0
            self.initialize_epoch(o)
            rewards = []
            log_probs = []
            values = []
            for _ in range(self.max_ep_len):
                log_prob, a, value = self.step(o)
                o2, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([r], dtype=torch.float, device=self.device))
                o = o2
                if d:
                    break
            running_reward = self.running_avg_rate * running_reward + (1 - self.running_avg_rate) * ep_ret
            ep_rewards.append(running_reward)

            returns_tensor = torch.cat(self.compute_returns(rewards)).detach()
            log_probs = torch.cat(log_probs)
            values = torch.cat(values)
            advantage = returns_tensor - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            loss_info = dict(actor_loss=actor_loss.detach().cpu().numpy(),
                             critic_loss=critic_loss.detach().cpu().numpy(),
                             ep_len=ep_len)
            self.logger.store(**loss_info)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            if ep % self.print_every == 0:
                self.logger.log_tabular('epoch', ep)
                self.logger.log_tabular('running_reward', running_reward)
                self.logger.log_tabular('ep_len', average_only=True)
                self.logger.log_tabular('actor_loss', average_only=True)
                self.logger.log_tabular('critic_loss', average_only=True)
                self.logger.log_tabular('time', time.time() - start_time)
                start_time = time.time()
                self.logger.dump_tabular()
        return ep_rewards
