import argparse
import torch
import numpy as np
import random
import time
import os
from matplotlib import pyplot as plt

from multiprocessing import Pool

from pomdp.agents.ac.base import ActorCritic
from pomdp.agents.ac.lstm_ac import LstmActorCritic
from pomdp.agents.ac.finite_block_controller import FscActorCritic
from pomdp.envs.cartpole import CartpoleEnv


def get_agent(env, name, seed):
    if name == 'lstm':
        agent = LstmActorCritic(
            env=env,
            gamma=args.gamma,
            seed=seed,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            print_every=args.print_every,
            running_avg_rate=args.running_avg_rate,
            data_dir=args.data_dir,
            h_dim=args.h_dim)
    elif name == 'ac':
        agent = ActorCritic(
            env=env,
            gamma=args.gamma,
            seed=seed,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            print_every=args.print_every,
            running_avg_rate=args.running_avg_rate,
            data_dir=args.data_dir)
    else:
        agent = FscActorCritic(
            env=env,
            gamma=args.gamma,
            seed=seed,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            print_every=args.print_every,
            running_avg_rate=args.running_avg_rate,
            data_dir=args.data_dir,
            max_hist_len=args.max_hist_len
        )
    return agent.name, agent


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def worker(seed, algo_name, par):
    print('Algo:{}, seed: {}, par: {} has started.'.format(algo_name, seed, par))
    args.max_hist_len = par
    st = time.time()
    env = CartpoleEnv()
    set_seed(seed)
    algo_name, agent = get_agent(env, algo_name, seed)
    res = agent.train(args.epochs)
    et = time.time()
    print('Algo: {}, seed: {} has finished, elapsed time: {:2.4f}s.'.format(algo_name, seed, et - st))
    return algo_name, res


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='cartpole')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epochs', type=int, default=15000)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=1e-4)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument("--max_hist_len", "--max_hist_len", type=int, default=5)
parser.add_argument('--running_avg_rate', type=float, default=0.99)
parser.add_argument("--data_dir", type=str, default='experiments')
parser.add_argument("--num_workers", "--num_workers", type=int, default=10)
args = parser.parse_args()

ALGOS = [('lstm', 1), ('ac', 1), ('fsc', 1), ('fsc', 2), ('fsc', 3)]
SEED_LIST = [1003, 727, 527, 714, 1225]
PLOT_NAME = 'env={}_g={}_ep={}_alr={}_clr={}_hdim={}_avg={}.png'.format(
    args.env_name, args.gamma, args.epochs, args.actor_lr, args.critic_lr, args.h_dim, args.running_avg_rate
)
records = {}
arguments = []
for algo, par in ALGOS:
    for seed in SEED_LIST:
        arguments.append([seed, algo, par])
with Pool(args.num_workers) as p:
    results = p.starmap(worker, arguments)

for (_, algo, _), (algo_name, record) in zip(arguments, results):
    if algo_name not in records:
        records[algo_name] = []
    records[algo_name].append(record)

for algo_name in records.keys():
    data = np.array(records[algo_name])
    y_mean = np.mean(data, axis=0)
    y_std = np.std(data, axis=0)
    x = np.arange(len(y_mean))
    plt.plot(x, y_mean)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.3)
plt.legend(records.keys())
plt.savefig(os.path.join('result', PLOT_NAME))


