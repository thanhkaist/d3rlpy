import os
import json
import time
import copy
from tqdm import tqdm

import torch
from torch import multiprocessing as mp

import numpy as np
from .utility import tensor
from .attackers import random_attack, critic_normal_attack, actor_mad_attack


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
    attack_stepsize = attack_epsilon / params.attack_iteration
    if rank == 0:
        print("[INFO] Using %s attack: eps=%f, n_iters=%d, sz=%f" %
              (params.attack_type.upper(), attack_epsilon, params.attack_iteration, attack_stepsize))

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
    for i in tqdm(range(n_trials), disable=(rank != 0), desc="{} attack".format(attack_type.upper())):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        state = env.reset()

        state = attack(state, attack_type, attack_epsilon, params.attack_iteration, attack_stepsize)
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict([state])[0]

            state, reward, done, _ = env.step(action)
            state = attack(state, attack_type, attack_epsilon, params.attack_iteration, attack_stepsize)
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


# def train_sarsa(algo, env, buffer=None, n_sarsa_steps=1000, n_warmups=1000):
#
#     logdir_sarsa = os.path.join(args.ckpt[:args.ckpt.rfind('/')], 'sarsa_model')
#     model_path = os.path.join(logdir_sarsa, 'sarsa_ntrains{}_warmup{}.pt'.format(n_sarsa_steps, n_warmups))
#
#
#     algo = make_bound_for_network(algo)
#
#     # We need to re-initialize the critic, not using the old one (following SA-DDPG)
#     algo._impl._q_func.reset_weight()
#     algo._impl._targ_q_func.reset_weight()
#
#     if not os.path.exists(logdir_sarsa):
#         os.mkdir(logdir_sarsa)
#     if os.path.exists(model_path):
#         print('Found pretrained SARSA: ', model_path)
#         algo.load_model(model_path)
#     else:
#         print('Not found pretrained SARSA: ', model_path)
#         algo.fit_sarsa(env, buffer, n_sarsa_steps, n_warmups)
#         algo.save_model(model_path)
#
#     return algo
