import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import itertools

from pomdp.agents.td3.base import Agent


class HistTd3Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, max_hist_len):
        super(HistTd3Critic, self).__init__()
        self.hist_linear1 = nn.Linear((obs_dim + act_dim) * max_hist_len, 128)
        self.hist_linear2 = nn.Linear(128, 128)
        self.cur_feature_linear1 = nn.Linear(obs_dim + act_dim, 128)
        self.cur_feature_linear2 = nn.Linear(128, 128)
        self.combined_linear1 = nn.Linear(128 + 128, 128)
        self.combined_linear2 = nn.Linear(128, 1)

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        x = torch.flatten(torch.cat([hist_obs, hist_act], dim=-1), start_dim=1)
        x = F.relu(self.hist_linear1(x))
        hist_out = F.relu(self.hist_linear2(x))

        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.cur_feature_linear1(x))
        x = F.relu(self.cur_feature_linear2(x))

        x = torch.cat([hist_out, x], dim=-1)
        x = F.relu(self.combined_linear1(x))
        x = self.combined_linear2(x)
        return torch.squeeze(x, -1)


class HistTd3Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, max_hist_len):
        super(HistTd3Actor, self).__init__()
        self.act_limit = act_limit
        self.hist_linear1 = nn.Linear((obs_dim + act_dim) * max_hist_len, 128)
        self.hist_linear2 = nn.Linear(128, 128)
        self.cur_feature_linear1 = nn.Linear(obs_dim, 128)
        self.cur_feature_linear2 = nn.Linear(128, 128)
        self.combined_linear1 = nn.Linear(128 + 128, 128)
        self.combined_linear2 = nn.Linear(128, act_dim)

    def forward(self, obs, hist_obs, hist_act, hist_seg_len, train=True):
        x = torch.flatten(torch.cat([hist_obs, hist_act], dim=-1), start_dim=1)
        x = F.relu(self.hist_linear1(x))
        hist_out = F.relu(self.hist_linear2(x))

        x = obs
        x = F.relu(self.cur_feature_linear1(x))
        x = F.relu(self.cur_feature_linear2(x))

        x = torch.cat([hist_out, x], dim=-1)
        x = F.relu(self.combined_linear1(x))
        x = torch.tanh(self.combined_linear2(x))
        return self.act_limit * x


class HistTd3ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, max_hist_len):
        super(HistTd3ActorCritic, self).__init__()
        self.q1 = HistTd3Critic(obs_dim, act_dim, max_hist_len)
        self.q2 = HistTd3Critic(obs_dim, act_dim, max_hist_len)
        self.pi = HistTd3Actor(obs_dim, act_dim, act_limit, max_hist_len)


class HistTd3(Agent):
    def __init__(
        self,
        env,
        test_env,
        seed,
        steps_per_epoch=4000,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        actor_lr=1e-3,
        critic_lr=1e-3,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        act_noise=0.1,
        target_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        num_test_episodes=10,
        max_ep_len=1000,
        batch_size=100,
        max_hist_len=100,
        running_avg_rate=0.95,
        data_dir=".",
    ):
        self.name = "Hist" + str(max_hist_len)
        super(HistTd3, self).__init__(
            env,
            test_env,
            seed,
            steps_per_epoch,
            replay_size,
            gamma,
            polyak,
            actor_lr,
            critic_lr,
            start_steps,
            update_after,
            update_every,
            act_noise,
            target_noise,
            noise_clip,
            policy_delay,
            num_test_episodes,
            max_ep_len,
            batch_size,
            max_hist_len,
            running_avg_rate,
            data_dir,
        )

    def get_agent(self):
        agent = HistTd3ActorCritic(
            self.obs_dim, self.act_dim, self.act_limit, self.max_hist_len
        )
        q_params = itertools.chain(agent.q1.parameters(), agent.q2.parameters())
        return agent, q_params

    def get_optimizers(self, actor_lr, critic_lr):
        pi_optimizer = Adam(self.ac.pi.parameters(), lr=actor_lr)
        q_optimizer = Adam(self.q_params, lr=critic_lr)
        return pi_optimizer, q_optimizer
