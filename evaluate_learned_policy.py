import argparse
import gym
import time
import copy
import os
import json

import d3rlpy

from torch import multiprocessing as mp

import pandas as pd
import numpy as np

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

IQM = lambda x: metrics.aggregate_iqm(x) # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0) # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)

from d3rlpy.preprocessing.scalers import StandardScaler
from d3rlpy.adversarial_training.utility import make_checkpoint_list, copy_file, EvalLogger
from d3rlpy.adversarial_training.eval_utility import (
    ENV_SEED,
    eval_clean_env,
    eval_env_under_attack,
    eval_multiprocess_wrapper,
    train_sarsa
)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n_eval_episodes', type=int, default=50)

SUPPORTED_TRANSFORMS = ['random', 'adversarial_training']
parser.add_argument('--transform', type=str, default='random', choices=SUPPORTED_TRANSFORMS)

SUPPORTED_ATTACKS = ['random', 'critic_normal', 'actor_mad', 'sarsa']
parser.add_argument('--attack_type', type=str, default='random', choices=SUPPORTED_ATTACKS)
parser.add_argument('--attack_epsilon', type=float, default=None)
parser.add_argument('--attack_type_list', type=str, default='random', nargs='+')
parser.add_argument('--attack_epsilon_list', type=float, default=1e-4, nargs='+')
parser.add_argument('--attack_iteration', type=int, default=0)
parser.add_argument('--no_clip', action='store_true')
parser.add_argument('--no_assert', action='store_true')

SUPPORTED_OPTIMS = ['pgd', 'sgld']
parser.add_argument('--optimizer', type=str, default='pgd', choices=SUPPORTED_OPTIMS)

parser.add_argument('--ckpt', type=str, default='.')
parser.add_argument('--n_seeds_want_to_test', type=int, default=1)
parser.add_argument('--ckpt_steps', type=str, default="model_500000.pt")

parser.add_argument('--disable_clean', action='store_true')
parser.add_argument('--mp', action='store_true')
parser.add_argument('--n_processes', type=int, default=5)

parser.add_argument('--eval_logdir', type=str, default='eval_results')

parser.add_argument('--online_rl', action='store_true')
args = parser.parse_args()


"""" Pre-defined constant for evaluation """
ATTACK_ITERATION=dict(
    random=1,
    critic_normal=5,
    actor_mad=5,
    sarsa=5
)


def eval_func(algo, env, writer, attack_type, attack_epsilon, params):
    multiprocessing = params.mp
    _args = copy.deepcopy(params)
    _args.attack_type = attack_type
    _args.attack_epsilon = attack_epsilon

    _args.attack_iteration = ATTACK_ITERATION[attack_type]

    unorm_score, norm_score, unorm_score_attack, norm_score_attack = None, None, None, None
    if multiprocessing:
        print("[INFO] Multiple-processing evaluating...")
        env_list = []
        env_list.append(env)
        for i in range(_args.n_processes - 1):
            _env = gym.make(_args.dataset)
            _env.seed(ENV_SEED)
            env_list.append(_env)
        if not _args.disable_clean:
            unorm_score = eval_multiprocess_wrapper(algo, eval_clean_env, env_list, _args)
        unorm_score_attack = eval_multiprocess_wrapper(algo, eval_env_under_attack, env_list, _args)

        del env_list

    else:
        print("[INFO] Normally evaluating...")
        # func_args = (0, algo, env, _args.seed, _args)  # algo, env, start_seed, args
        func_args = (0, algo, env, ENV_SEED, _args)  # algo, env, start_seed, args

        if not _args.disable_clean:
            unorm_score = eval_clean_env(func_args)
        unorm_score_attack = eval_env_under_attack(func_args)


    if not _args.disable_clean:
        norm_score = env.env.wrapped_env.get_normalized_score(unorm_score) * 100
        writer.log(attack_type="clean", attack_epsilon=attack_epsilon,
                   attack_iteration=_args.attack_iteration,
                   unorm_score=unorm_score, norm_score=norm_score)
    norm_score_attack = env.env.wrapped_env.get_normalized_score(unorm_score_attack) * 100

    writer.log(attack_type=attack_type, attack_epsilon=attack_epsilon,
               attack_iteration=_args.attack_iteration,
               unorm_score=unorm_score_attack, norm_score=norm_score_attack)

    print("***** Env: %s - method: %s *****" % (_args.dataset, _args.ckpt.split('/')[-3]))
    if unorm_score is not None:
        print("Clean env: unorm = %.3f, norm = %.2f" % (unorm_score, norm_score))
    print("Noise env: unorm = %.3f, norm = %.2f" % (unorm_score_attack, norm_score_attack))
    return unorm_score, norm_score, unorm_score_attack, norm_score_attack


def main(args):
    if not os.path.exists(args.eval_logdir):
        os.makedirs(args.eval_logdir)

    print("[INFO] Logging evalutation into: %s" % (args.eval_logdir))

    print("[INFO] Loading dataset: %s\n" % (args.dataset))
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    d3rlpy.seed(args.seed)
    env.seed(ENV_SEED)

    if args.online_rl:
        env_name = args.dataset.split('-')[0]

        dataset1, _ = d3rlpy.datasets.get_dataset('{}-random-v0'.format(env_name))
        dataset2, _ = d3rlpy.datasets.get_dataset('{}-medium-v0'.format(env_name))
        dataset3, _ = d3rlpy.datasets.get_dataset('{}-medium-replay-v0'.format(env_name))
        dataset4, _ = d3rlpy.datasets.get_dataset('{}-expert-v0'.format(env_name))

        dataset1.extend(dataset2)
        dataset1.extend(dataset3)
        dataset1.extend(dataset4)

        scaler = StandardScaler(dataset1)
        td3 = d3rlpy.algos.TD3PlusBC(scaler=scaler, use_gpu=args.gpu, env_name=args.dataset)

    else:
        ### Initialize algorithm
        td3 = d3rlpy.algos.TD3PlusBC(scaler="standard", use_gpu=args.gpu, env_name=args.dataset)

        ### Convert dataset to list of transitions to compute mean & std
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        td3._scaler.fit(transitions)  # Compute mean & std of dataset


    td3.build_with_env(env)  # Create policy/critic for env, must be performed after fitting scaler

    list_checkpoints = make_checkpoint_list(args.ckpt, args.n_seeds_want_to_test, args.ckpt_steps)

    print("[INFO] Evaluating %d checkpoint(s)\n" % (args.n_seeds_want_to_test))

    # Initialize writer for first checkpoint, and append next checkpoints
    writer = EvalLogger(ckpt=list_checkpoints[0], eval_logdir=args.eval_logdir,
                        prefix='eval_v1', eval_args=args)

    # Structure: NxRxC = N attack's types x R epsilon values x C seeds
    N = len(args.attack_type_list)
    R = len(args.attack_epsilon_list)
    C = args.n_seeds_want_to_test

    norm_scores = np.zeros((1, 1, C))
    norm_score_attacks = np.zeros((N, R, C))

    # Scan through all checkpoints
    n_seeds = args.n_seeds_want_to_test if \
        len(list_checkpoints) > args.n_seeds_want_to_test else len(list_checkpoints)
    for c, checkpoint in enumerate(list_checkpoints[:n_seeds]):
        if c > 0:
            # If only have 1 seed, do not write anything
            writer.init_info_from_ckpt(checkpoint)
            writer.write_header()

        td3.load_model(checkpoint)
        args.ckpt = checkpoint
        print("===> Eval checkpoint: %s" % (checkpoint))
        start = time.time()
        for n, attack_type in enumerate(args.attack_type_list):
            if attack_type in ['sarsa']:
                td3 = train_sarsa(td3, env, checkpoint)     # If there is checkpoint, load it

            for r, attack_epsilon in enumerate(args.attack_epsilon_list):
                args.disable_clean = not (r == 0) or not (n == 0)
                _, _norm_score, _, _norm_score_attack = \
                    eval_func(td3, env, writer, attack_type, attack_epsilon, args)

                if not args.disable_clean:
                    norm_scores[n, r, c] = _norm_score
                norm_score_attacks[n, r, c] = _norm_score_attack
        print("\n<=== Evaluation time for seed %d: %.3f (s)\n" % (c + 1, time.time() - start))


    writer.print("\n\n====================== Summary ======================\n")
    writer.print("Average clean: mean=%.2f, std=%.2f, median=%.2f, iqm=%.2f, og=%.2f (%d seeds)\n" %
                 (MEAN([norm_scores[0, 0]]), np.std(norm_scores, axis=2).squeeze(),
                  MEDIAN([norm_scores[0, 0]]),
                  IQM([norm_scores[0, 0]]), OG([norm_scores[0, 0]]),
                  n_seeds))

    columns = ["mean", "std", "median", "iqm", "og", "n_seeds"]
    data = [[MEAN([norm_scores[0, 0]]), np.std(norm_scores, axis=2).squeeze(),
             MEDIAN([norm_scores[0, 0]]), IQM([norm_scores[0, 0]]), OG([norm_scores[0, 0]]),
             n_seeds]]

    score_dict = {
        "env_name": args.dataset,
        "n_seeds": n_seeds,
        "clean": norm_scores[0, 0].tolist()
    }
    summary = pd.DataFrame(data, columns=columns, index=["clean"])
    for n in range(N):
        for r in range(R):
            writer.print("Attack: %15s [eps=%.4f]: mean=%.2f, std=%.2f, median=%.2f, iqm=%.2f, og=%.2f (%d seeds)\n" %
                         (args.attack_type_list[n], args.attack_epsilon_list[r],
                          MEAN([norm_score_attacks[n, r]]),
                          np.std(norm_score_attacks, axis=2)[n, r],
                          MEDIAN([norm_score_attacks[n, r]]),
                          IQM([norm_score_attacks[n, r]]),
                          OG([norm_score_attacks[n, r]]),
                          n_seeds))
            data = [[
                MEAN([norm_score_attacks[n, r]]),
                np.std(norm_score_attacks, axis=2)[n, r],
                MEDIAN([norm_score_attacks[n, r]]),
                IQM([norm_score_attacks[n, r]]),
                OG([norm_score_attacks[n, r]]),
                n_seeds
            ]]
            index = ["%15s-[eps=%.4f]" % (args.attack_type_list[n], args.attack_epsilon_list[r])]
            _summary = pd.DataFrame(data, columns=columns, index=index)
            summary = summary.append(_summary)

            score_dict.update({
                str(args.attack_type_list[n]) + '-' + str(args.attack_epsilon_list[r]): norm_score_attacks[n, r].tolist()
            })

    writer.close()
    pickle_filename = writer.logfile[:-3] + 'pkl'
    summary.to_pickle(pickle_filename)
    json_filename = writer.logfile[:-3] + 'json'

    with open(json_filename, 'w') as fp:
        json.dump(score_dict, fp, sort_keys=True)

    # Always maintain latest files
    copy_file(src=writer.logfile, des=writer.logfile[:-18] + 'latest.txt')
    copy_file(src=pickle_filename, des=pickle_filename[:-18] + 'latest.pkl')
    copy_file(src=json_filename, des=writer.logfile[:-18] + 'latest.json')



if __name__ == '__main__':
    if args.mp:
        mp.set_start_method("spawn")
    main(args)

