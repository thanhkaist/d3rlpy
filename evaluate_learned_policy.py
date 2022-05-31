import argparse
import d3rlpy
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp

import numpy as np

import gym
from tqdm import tqdm
import time
import copy
import os

from d3rlpy.models.torch.policies import WrapperBoundDeterministicPolicy
from d3rlpy.models.torch.q_functions.ensemble_q_function import WrapperBoundEnsembleContinuousQFunction
from d3rlpy.adversarial_training.attackers import critic_normal_attack, actor_mad_attack, random_attack
from d3rlpy.adversarial_training.utility import tensor, EvalLogger


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

parser.add_argument('--disable_clean', action='store_true')
parser.add_argument('--ckpt', type=str, default='.')
parser.add_argument('--mp', action='store_true')
parser.add_argument('--n_processes', type=int, default=5)

parser.add_argument('--eval_logdir', type=str, default='eval_results')
args = parser.parse_args()


"""
##### Functions used to evaluate
"""
def eval_clean_env(params):
    rank, algo, env, start_seed, params = params
    n_trials = params.n_eval_episodes

    episode_rewards = []
    for i in tqdm(range(n_trials), disable=(rank != 0)):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        state = env.reset()
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict([state])[0]

            state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

    unorm_score = float(np.mean(episode_rewards))
    return unorm_score


def eval_env_under_attack(params):
    rank, algo, env, start_seed, params = params
    n_trials = params.n_eval_episodes

    # Set seed
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    attack_type = params.attack_type
    attack_epsilon = params.attack_epsilon
    if attack_type in ['critic_normal']:
        attack_iteration = 5
    elif attack_type in ['actor_mad']:
        attack_iteration = 10
    else:
        attack_iteration = 1
    attack_stepsize = attack_epsilon / attack_iteration
    if rank == 0:
        print("[INFO] Using %s attack: eps=%f, n_iters=%d, sz=%f" %
              (params.attack_type.upper(), attack_epsilon, attack_iteration, attack_stepsize))

    def attack(state, type, attack_epsilon=None, attack_iteration=None, attack_stepsize=None):
        if type in ['random']:
            ori_state_tensor = tensor(state, algo._impl.device)
            perturb_state = random_attack(
                ori_state_tensor, attack_epsilon,
                algo._impl._obs_min, algo._impl._obs_max,
                algo.scaler
            )
            perturb_state = perturb_state.cpu().numpy()

        elif type in ['critic_normal']:
            ori_state_tensor = tensor(state, algo._impl.device)         # original, unnormalized
            perturb_state = critic_normal_attack(
                ori_state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min, algo._impl._obs_max,
                algo.scaler
            )
            perturb_state = perturb_state.cpu().numpy()

        elif type in ['actor_mad']:
            ori_state_tensor = tensor(state, algo._impl.device)         # original, unnormalized
            perturb_state = actor_mad_attack(
                ori_state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min, algo._impl._obs_max,
                algo.scaler
            )
            perturb_state = perturb_state.cpu().numpy()

        else:
            raise NotImplementedError
        return perturb_state.squeeze()

    episode_rewards = []
    for i in tqdm(range(n_trials), disable=(rank != 0)):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        state = env.reset()

        state = attack(state, attack_type, attack_epsilon, attack_iteration, attack_stepsize)
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict([state])[0]

            state, reward, done, _ = env.step(action)
            state = attack(state, attack_type, attack_epsilon, attack_iteration, attack_stepsize)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

    unorm_score = float(np.mean(episode_rewards))
    return unorm_score


def eval_multiprocess_wrapper(algo, func, env_list, params):
    n_trials_per_each = int(params.n_eval_episodes / params.n_processes)
    n_trials_for_last = n_trials_per_each if params.n_eval_episodes % params.n_processes == 0 else \
        n_trials_per_each + params.n_eval_episodes % params.n_processes

    args_list = []
    for i in range(params.n_processes):
        params_tmp = copy.deepcopy(params)

        if i == params_tmp.n_processes - 1:  # last iteration
            params_tmp.n_eval_episodes = n_trials_for_last
        else:
            params_tmp.n_eval_episodes = n_trials_per_each

        start_seed = n_trials_per_each * i + 1
        args_list.append((i, algo, env_list[i], start_seed, params_tmp))

    with mp.Pool(params.n_processes) as pool:
        unorm_score = pool.map(func, args_list)
        unorm_score = np.mean(unorm_score)

    return unorm_score


def make_bound_for_network(algo):
    # For convex relaxation: This is tested for TD3, not guarantee to work with other algorithms
    algo._impl._q_func = WrapperBoundEnsembleContinuousQFunction(
        q_func=algo._impl._q_func,
        observation_shape=algo.observation_shape[0],
        action_shape=algo.action_size,
        device=algo._impl.device
    )
    algo._impl._targ_q_func = WrapperBoundEnsembleContinuousQFunction(
        q_func=algo._impl._targ_q_func,
        observation_shape=algo.observation_shape[0],
        action_shape=algo.action_size,
        device=algo._impl.device
    )

    algo._impl._policy = WrapperBoundDeterministicPolicy(
        policy=algo._impl._policy,
        observation_shape=algo.observation_shape[0],
        device=algo._impl.device
    )
    algo._impl._targ_policy = WrapperBoundDeterministicPolicy(
        policy=algo._impl._targ_policy,
        observation_shape=algo.observation_shape[0],
        device=algo._impl.device
    )

    return algo


def train_sarsa(algo, env, buffer=None, n_sarsa_steps=1000, n_warmups=1000):

    logdir_sarsa = os.path.join(args.ckpt[:args.ckpt.rfind('/')], 'sarsa_model')
    model_path = os.path.join(logdir_sarsa, 'sarsa_ntrains{}_warmup{}.pt'.format(n_sarsa_steps, n_warmups))


    algo = make_bound_for_network(algo)

    # We need to re-initialize the critic, not using the old one (following SA-DDPG)
    algo._impl._q_func.reset_weight()
    algo._impl._targ_q_func.reset_weight()

    if not os.path.exists(logdir_sarsa):
        os.mkdir(logdir_sarsa)
    if os.path.exists(model_path):
        print('Found pretrained SARSA: ', model_path)
        algo.load_model(model_path)
    else:
        print('Not found pretrained SARSA: ', model_path)
        algo.fit_sarsa(env, buffer, n_sarsa_steps, n_warmups)
        algo.save_model(model_path)

    return algo


def main(args):
    if not os.path.exists(args.eval_logdir):
        os.makedirs(args.eval_logdir)

    writer = EvalLogger(args)

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    ### Initialize algorithm
    td3 = d3rlpy.algos.TD3PlusBC(scaler="standard", use_gpu=args.gpu, env_name=args.dataset)
    td3.build_with_env(env)  # Create policy/critic w.r.t. env
    td3.load_model(args.ckpt)


    ### Convert dataset to list of transitions to compute mean & std
    transitions = []
    for episode in dataset.episodes:
        transitions += episode.transitions

    td3._scaler.fit(transitions)  # Compute mean & std of dataset

    # if args.attack_type.startswith('sarsa'):
    #     td3 = train_sarsa(td3, env)


    def eval_func(attack_type, attack_epsilon, disable_clean):
        args_clone = copy.deepcopy(args)
        args_clone.attack_type = attack_type
        args_clone.attack_epsilon = attack_epsilon
        args_clone.disable_clean = disable_clean

        unorm_score, norm_score, unorm_score_noise, norm_score_noise = None, None, None, None
        if not args_clone.mp:
            print('[INFO] Normally evaluating...')
            start = time.time()
            func_args = (0, td3, env, args_clone.seed, args_clone)  # algo, env, start_seed, args

            if not args_clone.disable_clean:
                unorm_score = eval_clean_env(func_args)
            unorm_score_noise = eval_env_under_attack(func_args)

        else:
            print('[INFO] Multiple-processing evaluating...')
            start = time.time()
            env_list = []
            env_list.append(env)
            for i in range(args_clone.n_processes - 1):
                _env = gym.make(args_clone.dataset)
                _env.seed(args_clone.seed)
                env_list.append(_env)
            if not args_clone.disable_clean:
                unorm_score = eval_multiprocess_wrapper(td3, eval_clean_env, env_list, args_clone)
            unorm_score_noise = eval_multiprocess_wrapper(td3, eval_env_under_attack, env_list, args_clone)

            del env_list

        if not args_clone.disable_clean:
            norm_score = env.env.wrapped_env.get_normalized_score(unorm_score) * 100
            writer.log(attack_type="clean", attack_epsilon=attack_epsilon,
                       attack_iteration=args_clone.attack_iteration,
                       unorm_score=unorm_score, norm_score=norm_score)
        norm_score_noise = env.env.wrapped_env.get_normalized_score(unorm_score_noise) * 100

        writer.log(attack_type=attack_type, attack_epsilon=attack_epsilon,
                   attack_iteration=args_clone.attack_iteration,
                   unorm_score=unorm_score_noise, norm_score=norm_score_noise)

        print("***** Env: %s - method: %s *****" % (args_clone.dataset, args_clone.ckpt.split('/')[-3]))
        if unorm_score is not None:
            print("Clean env: unorm = %.3f, norm = %.2f" % (unorm_score, norm_score))
        print("Noise env: unorm = %.3f, norm = %.2f" % (unorm_score_noise, norm_score_noise))
        # print("=> Time(s) for evaluation: %.3f" % (time.time() - start))

    for attack_type in args.attack_type_list:
        for i, attack_epsilon in enumerate(args.attack_epsilon_list):
            if i == 0:
                disable_clean = args.disable_clean
            else:
                disable_clean = True

            eval_func(attack_type, attack_epsilon, disable_clean)

    writer.close()


if __name__ == '__main__':
    if args.mp:
        mp.set_start_method("spawn")
    main(args)

