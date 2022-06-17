import os
import numpy as np
from matplotlib import pyplot as plt


def read_col(file_name, idx, record_len):
    f = open(file_name, "r")
    first = True
    record = []
    for line in f.readlines():
        if first:
            first = False
            continue
        f_list = [float(i) for i in line.split("\t")]
        record.append(f_list[idx])
    record = record[:record_len]
    f.close()
    return record


def get_records(file_name, idx, records, algo, nb, seeds, record_len):
    if len(algo) == 5:
        algo = (
            "SBC(" + algo[-1] + ")" if algo[:4] == "Hist" else "RNN(" + algo[-1] + ")"
        )
    else:
        algo = "SBC-" + algo[6:] if algo[:4] == "Hist" else "RNN-" + algo[5:]
    for seed in seeds:
        cur_file_name = os.path.join(file_name, "s" + seed)
        if idx == 0:
            record = np.load(os.path.join(cur_file_name, "ret.npy"))
        else:
            record = read_col(
                os.path.join(cur_file_name, "progress.txt"), idx, record_len
            )
        record = list(np.convolve(record, np.ones(nb), "valid") / nb)
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
    plt.savefig(os.path.join("result", plot_name))
    plt.close()


def plot_env(env_name, args, plot_name, nb, seeds, record_len):
    records = {}
    arguments = ["Hist5", "RNN5"]
    for i in range(len(arguments)):
        for cur in args:
            algo = arguments[i] + "_" + cur
            file_name = os.path.join("experiments", arguments[i] + "_" + env_name + cur)
            records = get_records(file_name, 1, records, algo, nb, seeds, record_len)
        plot(records, plot_name + ".png")


def plot_vel(env_name, plot_name, nb, seeds, record_len):
    records = {}
    arguments = ["Hist1", "Hist2", "Hist5", "LSTM1", "LSTM2", "LSTM5"]
    for i in range(len(arguments)):
        algo = arguments[i]
        file_name = os.path.join("experiments", arguments[i] + "_" + env_name)
        records = get_records(file_name, 1, records, algo, nb, seeds, record_len)
    plot(records, plot_name + ".png")


if __name__ == "__main__":
    nb = 10
    seeds = ["527", "714", "1003", "727", "1225"]
    record_len = 400
    env_name = "HalfCheetah-fprob"
    fprobs = ["0.10", "0.20", "0.50"]
    plot_env(env_name, fprobs, env_name, nb, seeds, record_len)

    env_name = "HalfCheetah-rnoise"
    rnoise = ["0.10", "0.50"]
    plot_env(env_name, rnoise, env_name, nb, seeds, record_len)

    env_name = "HalfCheetah-vel"
    plot_vel(env_name, env_name, nb, seeds, record_len)

    seeds = ["527", "714", "1003"]
    record_len = 200
    env_name = "Ant-fprob"
    fprobs = ["0.10", "0.20", "0.50"]
    plot_env(env_name, fprobs, env_name, nb, seeds, record_len)

    env_name = "Ant-rnoise"
    rnoise = ["0.10", "0.50"]
    plot_env(env_name, rnoise, env_name, nb, seeds, record_len)
