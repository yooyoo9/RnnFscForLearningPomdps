import os

import numpy as np
import time
from copy import deepcopy
import torch

from pomdp.utils.replay_buffer import ReplayBuffer
from pomdp.utils.logger import EpochLogger, setup_logger_kwargs


class Agent:
    def __init__(self, env, test_env, seed, steps_per_epoch=4000, replay_size=int(1e6), gamma=0.99, polyak=0.995,
                 actor_lr=1e-3, critic_lr=1e-3, start_steps=10000, update_after=1000, update_every=50, act_noise=0.1,
                 target_noise=0.2, noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
                 batch_size=100, max_hist_len=100, running_avg_rate=0.95, data_dir='.'):
        if not hasattr(self, 'name'):
            self.name = 'BaseAgent'
        logger_kwargs = setup_logger_kwargs(self.name, env.name, seed, data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.env = env
        self.test_env = test_env
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.batch_size = batch_size
        self.max_hist_len = max_hist_len

        self.running_avg_rate = running_avg_rate
        self.data_dict = data_dir

        self.obs_dim = env.observation_space_n
        self.act_dim = env.action_space_n
        self.act_limit = env.action_space.high[0]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ac, self.q_params = self.get_agent()
        self.ac.to(self.device)
        self.ac_targ = deepcopy(self.ac).to(self.device)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, max_size=self.replay_size)

        self.pi_optimizer, self.q_optimizer = self.get_optimizers(actor_lr, critic_lr)

    def get_agent(self):
        raise NotImplementedError

    def get_optimizers(self, actor_lr, critic_lr):
        raise NotImplementedError

    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        h_o, h_a, h_o2, h_a2, h_o_len, h_o2_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data['hist_obs_len'], data['hist_obs2_len']

        q1 = self.ac.q1(o, a, h_o, h_a, h_o_len)
        q2 = self.ac.q2(o, a, h_o, h_a, h_o_len)

        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2, h_o2, h_a2, h_o2_len)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2, h_o2, h_a2, h_o2_len)
            q2_pi_targ  = self.ac_targ.q2(o2, a2, h_o2, h_a2, h_o2_len)

            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        loss_q = loss_q1 + loss_q2
        return loss_q

    def compute_loss_pi(self, data):
        o, h_o, h_a, h_o_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_obs_len']
        a = self.ac.pi(o, h_o, h_a, h_o_len)
        q1_pi = self.ac.q1(o, a, h_o, h_a, h_o_len)
        return -q1_pi.mean()

    def update(self, data, timer):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        self.logger.store(LossQ=loss_q.item())

        if timer % self.policy_delay == 0:
            for p in self.q_params:
                p.requires_grad = False
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()
            for p in self.q_params:
                p.requires_grad = True
            self.logger.store(LossPi=loss_pi.item())
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, o_buff, a_buff, o_buff_len, train):
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(self.device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(self.device)
        h_l = torch.tensor([o_buff_len]).float().to(self.device)
        with torch.no_grad():
            a = self.ac.pi(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(self.device),
                           h_o, h_a, h_l).cpu().numpy().reshape(self.act_dim)
        if train:
            a += self.act_noise * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        rewards = []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            rewards = []
            o_buff = np.zeros([self.max_hist_len, self.obs_dim])
            a_buff = np.zeros([self.max_hist_len, self.act_dim])
            o_buff[0, :] = o
            o_buff_len = 0
            while not (d or (ep_len == self.max_ep_len)):
                a = self.get_action(o, o_buff, a_buff, o_buff_len, train=False)
                o2, r, d, _ = self.test_env.step(a)
                ep_ret += r
                rewards.append(r)
                ep_len += 1
                if o_buff_len == self.max_hist_len:
                    o_buff[:self.max_hist_len - 1] = o_buff[1:]
                    a_buff[:self.max_hist_len - 1] = a_buff[1:]
                    o_buff[self.max_hist_len - 1] = list(o)
                    a_buff[self.max_hist_len - 1] = list(a)
                else:
                    o_buff[o_buff_len + 1 - 1] = list(o)
                    a_buff[o_buff_len + 1 - 1] = list(a)
                    o_buff_len += 1
                o = o2
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        return rewards

    def save_model(self):
        fpath = self.logger.output_dir
        os.makedirs(fpath, exist_ok=True)
        model_fname = 'model.pt'
        model_elements = {'ac_state_dict': self.ac.state_dict(),
                          'target_ac_state_dict': self.ac_targ.state_dict(),
                          'pi_optimizer_state_dict': self.pi_optimizer.state_dict(),
                          'q_optimizer_state_dict': self.q_optimizer.state_dict()}
        model_fname = os.path.join(fpath, model_fname)
        torch.save(model_elements, model_fname)

    def train(self, epochs):
        start_time = time.time()
        t = 0
        while t < epochs * self.steps_per_epoch:
            o, ep_ret, ep_len = self.env.reset(), 0, 0
            o_buff = np.zeros([self.max_hist_len, self.obs_dim])
            a_buff = np.zeros([self.max_hist_len, self.act_dim])
            o_buff[0, :] = o
            o_buff_len = 0
            while True:
                if t > self.start_steps:
                    a = self.get_action(o, o_buff, a_buff, o_buff_len, train=True)
                else:
                    a = self.env.action_space.sample()
                o2, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1
                d = False if ep_len == self.max_ep_len else d
                self.replay_buffer.store(o, a, r, o2, d)
                if o_buff_len == self.max_hist_len:
                    o_buff[:self.max_hist_len - 1] = o_buff[1:]
                    a_buff[:self.max_hist_len - 1] = a_buff[1:]
                    o_buff[self.max_hist_len - 1] = list(o)
                    a_buff[self.max_hist_len - 1] = list(a)
                else:
                    o_buff[o_buff_len + 1 - 1] = list(o)
                    a_buff[o_buff_len + 1 - 1] = list(a)
                    o_buff_len += 1
                o = o2

                if t >= self.update_after and t % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch_with_history(self.batch_size, self.max_hist_len)
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        self.update(data=batch, timer=j)

                if (t + 1) % self.steps_per_epoch == 0:
                    ep = (t+1) // self.steps_per_epoch
                    returns = self.test_agent()
                    np.save(os.path.join(self.logger.output_dir, 'ret.npy'), np.array(returns))
                    self.logger.log_tabular('Epoch', ep)
                    self.logger.log_tabular('EpRet', average_only=True)
                    self.logger.log_tabular('TestEpRet', average_only=True)
                    self.logger.log_tabular('EpLen', average_only=True)
                    self.logger.log_tabular('TestEpLen', average_only=True)
                    self.logger.log_tabular('Time', time.time() - start_time)
                    self.logger.dump_tabular()
                t += 1
                if d or (ep_len == self.max_ep_len):
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    self.save_model()
                    break

