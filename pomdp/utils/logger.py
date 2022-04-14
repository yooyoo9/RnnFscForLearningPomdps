import json
import numpy as np
import os.path as osp, time, atexit, os


def statistics_scalar(x, with_min_and_max=False):
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = x.sum(), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean) ** 2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std


def setup_logger_kwargs(exp_name, env_name, seed=None, data_dir=None):
    # Make base path
    # ymd_time = time.strftime("%Y-%m-%d_")
    relpath = "".join([exp_name, "_", env_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        subfolder = "".join(["s", str(seed)])
        relpath = osp.join(relpath, subfolder)

    logger_kwargs = dict(
        output_dir=osp.join(data_dir, relpath), exp_name=exp_name, env_name=env_name
    )
    return logger_kwargs


class Logger:
    def __init__(
        self, output_dir, output_fname="progress.txt", exp_name=None, env_name=None
    ):
        self.output_dir = output_dir
        if not osp.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_file_path = osp.join(self.output_dir, output_fname)
        open_mode = "w"
        self.first_row = True
        self.log_headers = []
        self.output_file = open(self.output_file_path, open_mode)
        atexit.register(self.output_file.close)
        print("Logging data to %s" % self.output_file.name)
        self.log_current_row = {}
        self.exp_name = exp_name

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        self.log_current_row[key] = val

    def save_config(self, config):
        output = json.dumps(
            config,
            default=lambda o: "<not serializable>",
            separators=(",", ":\t\t"),
            indent=4,
            sort_keys=True,
        )
        with open(osp.join(self.output_dir, "config.json"), "w") as out:
            out.write(output)

    def dump_tabular(self):
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        # print("-" * n_slashes)
        # cur_str = ""
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            # print(fmt % (key, valstr))
            vals.append(valstr)
        if self.first_row:
            print("\t".join(self.log_headers))
        print("\t".join(map(str, vals)))
        # print(cur_str)
        # print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = (
                np.concatenate(v)
                if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0
                else v
            )
            stats = statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else "Average" + key, stats[0])
            if not (average_only):
                super().log_tabular("Std" + key, stats[1])
            if with_min_and_max:
                super().log_tabular("Max" + key, stats[3])
                super().log_tabular("Min" + key, stats[2])
        self.epoch_dict[key] = []
