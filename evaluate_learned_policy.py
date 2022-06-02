import argparse
import d3rlpy
from sklearn.model_selection import train_test_split

from torch import multiprocessing as mp

import numpy as np

import gym
from tqdm import tqdm
import time
import copy
import os


from d3rlpy.adversarial_training.utility import make_checkpoint_list, EvalLogger
from d3rlpy.adversarial_training.eval_utility import (
    eval_clean_env,
    eval_env_under_attack,
    eval_multiprocess_wrapper
)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n_eval_episodes', type=int, default=100)

SUPPORTED_TRANSFORMS = ['random', 'adversarial_training']
parser.add_argument('--transform', type=str, default='random', choices=SUPPORTED_TRANSFORMS)

SUPPORTED_ATTACKS = ['random', 'critic_normal', 'actor_mad']
parser.add_argument('--attack_type', type=str, default='random', choices=SUPPORTED_ATTACKS)
parser.add_argument('--attack_epsilon', type=float, default=None)
parser.add_argument('--attack_type_list', type=str, default='random', nargs='+')
parser.add_argument('--attack_epsilon_list', type=float, default=1e-4, nargs='+')
parser.add_argument('--attack_iteration', type=int, default=10)

parser.add_argument('--ckpt', type=str, default='.')
parser.add_argument('--n_seeds_want_to_test', type=int, default=1)
parser.add_argument('--ckpt_steps', type=str, default="model_500000.pt")

parser.add_argument('--disable_clean', action='store_true')
parser.add_argument('--mp', action='store_true')
parser.add_argument('--n_processes', type=int, default=5)

parser.add_argument('--eval_logdir', type=str, default='eval_results')
args = parser.parse_args()


"""" Pre-defined constant for evaluation """
ATTACK_ITERATION=dict(
    random=1,
    critic_normal=5,
    actor_mad=5
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
            _env.seed(_args.seed)
            env_list.append(_env)
        if not _args.disable_clean:
            unorm_score = eval_multiprocess_wrapper(algo, eval_clean_env, env_list, _args)
        unorm_score_attack = eval_multiprocess_wrapper(algo, eval_env_under_attack, env_list, _args)

        del env_list

    else:
        print("[INFO] Normally evaluating...")
        func_args = (0, algo, env, _args.seed, _args)  # algo, env, start_seed, args

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

    print("[INFO] Logging evalutation into: %s\n" % (args.eval_logdir))

    print("[INFO] Loading dataset: %s\n" % (args.dataset))
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    ### Initialize algorithm
    td3 = d3rlpy.algos.TD3PlusBC(scaler="standard", use_gpu=args.gpu, env_name=args.dataset)
    td3.build_with_env(env)  # Create policy/critic w.r.t. env

    ### Convert dataset to list of transitions to compute mean & std
    transitions = []
    for episode in dataset.episodes:
        transitions += episode.transitions
    td3._scaler.fit(transitions)  # Compute mean & std of dataset


    list_checkpoints = make_checkpoint_list(args.ckpt, args.n_seeds_want_to_test, args.ckpt_steps)

    print("[INFO] Evaluating %d checkpoint(s)\n" % (args.n_seeds_want_to_test))

    # Initialize writer for first checkpoint, and append next checkpoints
    writer = EvalLogger(ckpt=list_checkpoints[0], eval_logdir=args.eval_logdir)

    # Scan through all checkpoints
    norm_scores = np.zeros((len(args.attack_type), 1, args.n_seeds_want_to_test))

    # Structure: NxRxC = N attack's types x R epsilon values x C seeds
    N = len(args.attack_type)
    R = len(args.attack_epsilon_list)
    C = args.n_seeds_want_to_test
    norm_score_attacks = np.zeros((N, R, C))

    for c, checkpoint in enumerate(list_checkpoints[:args.n_seeds_want_to_test]):
        print("==> Evaluate checkpoint: %s" % (checkpoint))
        td3.load_model(checkpoint)
        start = time.time()
        for n, attack_type in enumerate(args.attack_type_list):
            for r, attack_epsilon in enumerate(args.attack_epsilon_list):
                args.disable_clean = not (r == 0)   # Only do clean test for first epsilon
                _, _norm_score, _, _norm_score_attack = \
                    eval_func(td3, env, writer, attack_type, attack_epsilon, args)

                if not args.disable_clean:
                    norm_scores[n, r, c] = _norm_score
                norm_score_attacks[n, r, c] = _norm_score_attack
        print("<== Evaluation time for 1 seed: %.3f\n" % (time.time() - start))


    writer.print("====================== Summary ======================\n")
    writer.print("Average clean: mean=%.2f, std=%.2f, median=%.2f (%d seeds)\n" %
                 (np.mean(norm_scores, axis=2).squeeze(), np.mean(norm_scores, axis=2).squeeze(),
                  np.median(norm_scores, axis=2).squeeze(), len(list_checkpoints)))
    for n in range(N):
        for r in range(R):
            writer.print("Average attack: %15s-eps=%.4f: mean=%.2f, std=%.2f, median=%.2f (%d seeds)\n" %
                         (args.attack_type_list[n], args.attack_epsilon_list[r],
                          np.mean(norm_score_attacks, axis=2).squeeze(),
                          np.mean(norm_score_attacks, axis=2).squeeze(),
                          np.median(norm_score_attacks, axis=2).squeeze(), len(list_checkpoints)))
    writer.close()

if __name__ == '__main__':
    if args.mp:
        mp.set_start_method("spawn")
    main(args)

