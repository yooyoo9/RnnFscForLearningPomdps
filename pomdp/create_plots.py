import os
import numpy as np
from matplotlib import pyplot as plt

seeds = ['527', '714', '1003', '1225', '727']


def read_col(file_name, idx):
    # idx1 for training error, idx2 for test error
    f = open(file_name, 'r')
    first = True
    record = []
    for line in f.readlines():
        if first:
            first = False
            continue
        f_list = [float(i) for i in line.split("\t")]
        record.append(f_list[idx])
    record = record
    f.close()
    return record


def get_records(file_name, idx, records, algo, nb):
    for seed in seeds:
        cur_file_name = os.path.join(file_name, 's' + seed)
        if idx == 0:
            record = np.load(os.path.join(cur_file_name, 'ret.npy'))
        else:
            record = read_col(os.path.join(cur_file_name, 'progress.txt'), idx)
        record = list(np.convolve(record, np.ones(nb), 'valid') / nb)
        if algo not in records:
            records[algo] = []
        records[algo].append(record)
    return records


def plot(records, plot_name):
    plt.figure()
    for algo in records.keys():
        data = np.array(records[algo])
        y_mean = np.mean(data, axis=0)
        y_std = np.std(data, axis=0)
        x = np.arange(len(y_mean))
        plt.plot(x, y_mean)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, interpolate=True, alpha=0.1)
    plt.legend(records.keys())
    plt.savefig(os.path.join('result', plot_name))
    plt.close()


def plot_env(env_name, args, plot_name, nb=10):
    for idx in range(3):
        records = {}
        arguments = ['LSTM5', 'Hist5']
        for i in range(len(arguments)):
            for cur in args:
                algo = arguments[i] + '_' + cur
                file_name = os.path.join('experiments', arguments[i] + '_' + env_name + cur)
                records = get_records(file_name, idx, records, algo, nb)
        if idx == 0:
            plot(records, plot_name + '-ret.png')
        elif idx == 1:
            plot(records, plot_name + '.png')
        else:
            plot(records, plot_name + '-test.png')


def plot_vel(env_name, plot_name, nb=10):
    # for idx in range(3):
    for idx in [1]:
        records = {}
        arguments = ['LSTM1', 'LSTM2', 'LSTM5', 'Hist1', 'Hist2', 'Hist5']
        for i in range(len(arguments)):
            algo = arguments[i]
            file_name = os.path.join('experiments', arguments[i] + '_' + env_name)
            records = get_records(file_name, idx, records, algo, nb)
        if idx == 0:
            plot(records, plot_name + '-ret.png')
        elif idx == 1:
            plot(records, plot_name + '.png')
        else:
            plot(records, plot_name + '-test.png')


def plot_agent(env_name, args, plot_name, nb=10):
    for idx in range(3):
        records = {}
        agent_name = 'LSTM_lr'
        for cur in args:
            algo = agent_name + '_' + cur
            file_name = os.path.join('experiments', agent_name + cur + '_' + env_name)
            records = get_records(file_name, idx, records, algo, nb)
        if idx == 0:
            plot(records, plot_name + '-ret.png')
        elif idx == 1:
            plot(records, plot_name + '.png')
        else:
            plot(records, plot_name + '-test.png')


if __name__ == "__main__":
    env_name = 'HalfCheetah-fprob'
    fprobs = ['0.10', '0.20', '0.50']
    plot_env(env_name, fprobs, env_name)

    env_name = 'HalfCheetah-rnoise'
    rnoise = ['0.10', '0.50']
    plot_env(env_name, rnoise, env_name)

    env_name = 'HalfCheetah-vel'
    args = ['0.01', '0.001', '0.0001']
    plot_name = 'HalfCheetah-vel-lr'
    plot_agent(env_name, args, plot_name)

    env_name = 'HalfCheetah-vel'
    plot_vel(env_name, env_name, nb=1)
