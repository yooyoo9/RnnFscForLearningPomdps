import argparse
import torch
import numpy as np
import random
import time

from multiprocessing import Pool

from pomdp.envs.pomdp_wrapper import ModifiedEnv
from pomdp.agents.td3.td3_rnn import RnnTd3
from pomdp.agents.td3.td3_hist import HistTd3


def get_agent(env, test_env, name, seed):
    if name == "rnn_td3":
        agent = RnnTd3(
            env=env,
            test_env=test_env,
            seed=seed,
            running_avg_rate=args.running_avg_rate,
            data_dir=args.data_dir,
            max_hist_len=args.max_hist_len,
        )
    else:
        agent = HistTd3(
            env=env,
            test_env=test_env,
            seed=seed,
            running_avg_rate=args.running_avg_rate,
            data_dir=args.data_dir,
            max_hist_len=args.max_hist_len,
        )
    return agent.name, agent


def set_seed(env, test_env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    test_env.reset(seed=seed)


def worker(seed, algo_name, par):
    print("Algo:{}, seed: {}, par: {} has started.".format(algo_name, seed, par))
    args.max_hist_len = par
    st = time.time()
    env = ModifiedEnv(args.env_name, args.env_type, args.fprob, args.rnoise)
    test_env = ModifiedEnv(args.env_name, args.env_type, args.fprob, args.rnoise)
    set_seed(env, test_env, seed)
    algo_name, agent = get_agent(env, test_env, algo_name, seed)
    res = agent.train(args.epochs)
    et = time.time()
    print(
        "Algo: {}, seed: {} has finished, elapsed time: {:2.4f}s.".format(
            algo_name, seed, et - st
        )
    )
    return algo_name, res


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="halfcheetah")
parser.add_argument("--env_type", type=str, default="remove_velocity")
parser.add_argument("--fprob", type=float, default=0.1)
parser.add_argument("--rnoise", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_hist_len", "--max_hist_len", type=int, default=5)
parser.add_argument("--running_avg_rate", type=float, default=0.99)
parser.add_argument(
    "--exp_name",
    type=str,
    default="-1",
    choices=["-1", "lstm", "hist_td3", "fsc", "lstm_td3", "rnn_td3"],
)
parser.add_argument("--data_dir", type=str, default="experiments")
parser.add_argument("--num_workers", "--num_workers", type=int, default=5)
args = parser.parse_args()

if args.exp_name == "-1":
    ALGOS = [("rnn_td3", args.max_hist_len)]
    SEED_LIST = [1003, 727, 527, 714, 1225]
    records = {}
    arguments = []
    for algo, par in ALGOS:
        for seed in SEED_LIST:
            arguments.append([seed, algo, par])
    with Pool(args.num_workers) as p:
        p.starmap(worker, arguments)
else:
    worker(args.seed, args.exp_name, args.max_hist_len)
