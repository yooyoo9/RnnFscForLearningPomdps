import argparse
import torch
import numpy as np
import random
import time
import os
from matplotlib import pyplot as plt

from multiprocessing import Pool

from pomdp.envs.halfcheetah import HalfCheetahEnv
from pomdp.agents.td3.td3_lstm import LstmTd3
from pomdp.agents.td3.td3_hist import HistTd3


def get_agent(env, name, seed):
    if name == 'lstm_td3':
        agent = LstmTd3(
            env=env,
            seed=seed,
            running_avg_rate=args.running_avg_rate,
            data_dir=args.data_dir,
            max_hist_len=args.max_hist_len
        )
    else:
        agent = HistTd3(
            env=env,
            seed=seed,
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
    env = HalfCheetahEnv(args.env_type, args.fprob, args.rnoise)
    set_seed(seed)
    algo_name, agent = get_agent(env, algo_name, seed)
    res = agent.train(args.epochs)
    et = time.time()
    print('Algo: {}, seed: {} has finished, elapsed time: {:2.4f}s.'.format(algo_name, seed, et - st))
    return algo_name, res


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='halfcheetah')
parser.add_argument('--env_type', type=str, default='flickering')
parser.add_argument('--fprob', type=float, default=0.1)
parser.add_argument('--rnoise', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--max_hist_len", "--max_hist_len", type=int, default=10)
parser.add_argument('--running_avg_rate', type=float, default=0.99)
parser.add_argument('--exp_name', type=str, default='-1', choices=['-1', 'lstm', 'hist_td3', 'fsc', 'lstm_td3'])
parser.add_argument("--data_dir", type=str, default='experiments')
parser.add_argument("--num_workers", "--num_workers", type=int, default=4)
args = parser.parse_args()

args.env_name = args.env_name + '-'
if args.env_type == 'remove_velocity':
    args.env_name += 'vel'
elif args.env_type == 'flickering':
    args.env_name += f'fprob{args.fprob:.2f}'
elif args.env_type == 'random_noise':
    args.env_name += f'rnoise{args.rnoise:.2f}'

if args.exp_name == '-1':
    ALGOS = [('hist_td3', 5)]
    # SEED_LIST = [1003, 727, 527, 714, 1225]
    SEED_LIST = [1003, 727, 527, 714, 1225]
    # PLOT_NAME = 'env={}_ep={}_avg={}10.png'.format(
    #     args.env_name, args.epochs, args.running_avg_rate
    # )
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

    # for algo_name in records.keys():
    #     data = np.array(records[algo_name])
    #     y_mean = np.mean(data, axis=0)
    #     y_std = np.std(data, axis=0)
    #     x = np.arange(len(y_mean))
    #     plt.plot(x, y_mean)
    #     plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.3)
    # plt.legend(records.keys())
    # plt.savefig(os.path.join('result', PLOT_NAME))
else:
    worker(args.seed, args.exp_name, args.max_hist_len)


