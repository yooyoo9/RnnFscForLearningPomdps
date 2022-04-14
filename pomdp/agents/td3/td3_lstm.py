from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import itertools

from pomdp.agents.td3.base import Agent


class LstmTd3Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(LstmTd3Critic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.hist_linear1 = nn.Linear(obs_dim, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.cur_feature_linear1 = nn.Linear(obs_dim + act_dim, 128)
        self.cur_feature_linear2 = nn.Linear(128, 128)
        self.combined_linear1 = nn.Linear(128 + 128, 128)
        self.combined_linear2 = nn.Linear(128, 1)

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1

        x = hist_obs
        x = F.relu(self.hist_linear1(x))
        x, _ = self.lstm(x)
        hist_out = torch.gather(
            x, 1, (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, 128).unsqueeze(1).long()
        ).squeeze(1)

        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.cur_feature_linear1(x))
        x = F.relu(self.cur_feature_linear2(x))

        x = torch.cat([hist_out, x], dim=-1)
        x = F.relu(self.combined_linear1(x))
        x = self.combined_linear2(x)
        return torch.squeeze(x, -1)


class LstmTd3Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super(LstmTd3Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.hist_linear1 = nn.Linear(obs_dim + act_dim, 256)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.cur_feature_linear1 = nn.Linear(obs_dim, 128)
        self.cur_feature_linear2 = nn.Linear(128, 128)
        self.combined_linear1 = nn.Linear(128 + 128, 128)
        self.combined_linear2 = nn.Linear(128, act_dim)

    def forward(self, obs, hist_obs, hist_act, hist_seg_len):
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1

        x = torch.cat([hist_obs, hist_act], dim=-1)
        x = F.relu(self.hist_linear1(x))
        x, _ = self.lstm(x)
        hist_out = torch.gather(
            x, 1, (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, 128).unsqueeze(1).long()
        ).squeeze(1)

        x = obs
        x = F.relu(self.cur_feature_linear1(x))
        x = F.relu(self.cur_feature_linear2(x))

        x = torch.cat([hist_out, x], dim=-1)
        x = F.relu(self.combined_linear1(x))
        x = torch.tanh(self.combined_linear2(x))
        return self.act_limit * x


class LstmTd3ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit=1):
        super(LstmTd3ActorCritic, self).__init__()
        self.q1 = LstmTd3Critic(obs_dim, act_dim)
        self.q2 = LstmTd3Critic(obs_dim, act_dim)
        self.pi = LstmTd3Actor(obs_dim, act_dim, act_limit)


class LstmTd3(Agent):
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
        self.name = "LSTM" + str(max_hist_len)
        # self.name = 'LSTM' + '_lr' + str(actor_lr)
        super(LstmTd3, self).__init__(
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
        agent = LstmTd3ActorCritic(self.obs_dim, self.act_dim, self.act_limit)
        q_params = itertools.chain(agent.q1.parameters(), agent.q2.parameters())
        return agent, q_params

    def get_optimizers(self, actor_lr, critic_lr):
        pi_optimizer = Adam(self.ac.pi.parameters(), lr=actor_lr)
        q_optimizer = Adam(self.q_params, lr=critic_lr)
        return pi_optimizer, q_optimizer
