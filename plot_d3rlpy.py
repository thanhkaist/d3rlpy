import argparse
import ast
import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob2
import pandas as pd
from d4rl import infos
import gym


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, nargs='+')
parser.add_argument('--radius', type=int, default=0)
parser.add_argument('--range', type=int, default=-1, help='Number of transitions want to plot')
parser.add_argument('--legend', type=str, default='', nargs='+')
parser.add_argument('--title', type=str, default='')
parser.add_argument('--shaded_std', type=bool, default=True)
parser.add_argument('--shaded_err', type=bool, default=False)
parser.add_argument('--train_test', action='store_true')

parser.add_argument('--env', type=str, default='cheetah_run')
parser.add_argument('--plot_as', type=str, choices=['epoch', 'step'], default='step')
parser.add_argument('--as_env', action='store_true')
parser.add_argument('--normalized', type=bool, default=True)

args = parser.parse_args()




def smooth(y, radius, mode='two_sided', valid_only=False):
    # Copy from: https://github.com/openai/baselines/blob/master/baselines/common/plot_util.py
    '''
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)
        else:
            padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
            x_padded = np.concatenate([x, padding], axis=0)
            assert x_padded.shape[1:] == x.shape[1:]
            assert x_padded.shape[0] == maxlen
            padded_xs.append(x_padded)
    return np.array(padded_xs)


def get_info_env(path):
    # Get information from json file
    info = dict(
        algorithm=None,
    )
    json_file = os.path.join(path, 'params.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    for k in info.keys():
        info[k] = data[k]

    # Get name of environment
    env_name = None
    for env in list(infos.DATASET_URLS.keys()):
        env_wo_version = env[:-1]
        if env_wo_version in path:
            env_name = env_wo_version
            break
    idx_envname = path.find(env_name)
    env_name = path[idx_envname:idx_envname + len(env_name) + 1]
    info['env_name'] = env_name
    return info


def get_data_in_subdir(parent_path):
    file_target = 'environment.csv'
    child_paths = [os.path.abspath(os.path.join(path, '..'))
                   for path in glob2.glob(os.path.join(parent_path, '**', file_target))]

    data_x_epoch, data_x_step,  data_y = [], [], []
    for path in child_paths:
        file_csv = os.path.join(path, file_target)
        data = []
        file = open(file_csv)
        csvreader = csv.reader(file)
        for line in csvreader:
            data.append(line)

        len_data = len(data)
        x_step, x_epoch, y = [], [], []
        for i in range(len_data):
            x_epoch.append(float(data[i][0]))
            x_step.append(float(data[i][1]))
            y.append(float(data[i][2]))

        x_epoch = np.array(x_epoch)
        x_step = np.array(x_step)
        y = np.array(y)
        y = smooth(y, radius=args.radius)

        data_x_epoch.append(x_epoch)
        data_x_step.append(x_step)
        data_y.append(y)

    info_env = get_info_env(child_paths[0])
    task_name = info_env['env_name']

    if args.plot_as in ['epoch']:
        return (data_x_epoch, data_y), task_name, info_env
    elif args.plot_as in ['step']:
        return (data_x_step, data_y), task_name, info_env
    else:
        raise NotImplementedError


def get_values_with_range(xs, ys, truncate):
    n_experiments = len(xs)
    _xs = []
    _ys = []
    for k in range(n_experiments):
        found_idxes = np.argwhere(xs[k] >= truncate)
        if len(found_idxes) == 0:
            print("[WARNING] Last index is {}, consider choose smaller range in {}".format(
                xs[k][-1], directories[i]))
            _xs.append(xs[k][:])
            _ys.append(ys[k][:])
        else:
            range_idx = found_idxes[0, 0]
            _xs.append(xs[k][:range_idx])
            _ys.append(ys[k][:range_idx])
    return _xs, _ys


def plot_multiple_results(directories):

    collect_data, plot_titles, info_envs = [], [], []
    for directory in directories:
        data_in_subdir, task_name, info_env = get_data_in_subdir(directory)

        collect_data.append(data_in_subdir)
        plot_titles.append(task_name)
        info_envs.append(info_env)

    env = gym.make(info_env['env_name'])

    # Plot data.
    results_mean, results_std = [], []
    for i in range(len(collect_data)):
        xs, ys = collect_data[i]
        if args.normalized:
            for k in range(len(ys)):
                ys[k] = env.get_normalized_score(ys[k]) * 100
        n_experiments = len(xs)

        for exp_i in range(n_experiments):
            xs[exp_i] = xs[exp_i]

        if args.range != -1:
            xs, ys = get_values_with_range(xs, ys, args.range)
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape

        usex = xs[0]
        ymean = np.nanmean(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)

        results_mean.append(ymean[-1])
        results_std.append(ystd[-1])

        ystderr = ystd / np.sqrt(len(ys))
        plt.plot(usex, ymean, label='config')
        # if args.shaded_err:
        #     plt.fill_between(usex, ymean - ystderr, ymean + ystderr, alpha=0.4)
        if args.shaded_std:
            plt.fill_between(usex, ymean - ystd, ymean + ystd, alpha=0.2)

    if args.title == '':
        plt.title(plot_titles[0], fontsize='x-large')
    else:
        plt.title(args.title, fontsize='x-large')
    plt.xlabel('Number of env steps', fontsize='x-large')
    plt.ylabel('Episode Return', fontsize='x-large')

    plt.tight_layout()

    if args.as_env:
        legend_name = [directories[i].split('/')[-2] for i in range(len(directories))]
    else:
        legend_name = [directories[i].split('/')[-1] for i in range(len(directories))]
    if args.legend != '':
        assert len(args.legend) == len(
            directories), "Provided legend is not match with number of directories"
        legend_name = args.legend

    plt.legend(legend_name, loc='best', fontsize=10)

    print("\t\t\t*********** Final results ***********")
    for i in range(len(legend_name)):
        print("%40s: Unormalized: mean = %.3f, std = %.3f / Normalized: mean = %.3f, std = %.3f" %
              (legend_name[i], results_mean[i], results_std[i],
               env.get_normalized_score(results_mean[i]) * 100,
               env.get_normalized_score(results_std[i]) * 100 ))

    plt.show()


if __name__ == '__main__':
    directories = []
    for i in range(len(args.dir)):
        if args.dir[i][-1] == '/':
            directories.append(args.dir[i][:-1])
        else:
            directories.append(args.dir[i])
    plot_multiple_results(directories)
