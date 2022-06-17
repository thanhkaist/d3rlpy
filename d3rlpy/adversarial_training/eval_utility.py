import os
import json
import pdb
import time
import copy
from tqdm import tqdm

import torch
from torch import multiprocessing as mp, true_divide

import numpy as np
from .utility import tensor
from .attackers import random_attack, critic_normal_attack, actor_mad_attack

DEBUG= False

def make_sure_type_is_float32(x):
    assert isinstance(x, np.ndarray)
    x = x.astype(np.float32) if x.dtype == np.float64 else x
    assert x.dtype == np.float32
    return x

"""
##### Functions used to evaluate
"""
def eval_clean_env(params):
    print("EVALUATING ON CLEAN EVALUATION")
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
            state = make_sure_type_is_float32(state)
            action = algo.predict([state])[0]

            state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

    unorm_score = float(np.mean(episode_rewards))
    return unorm_score


def eval_env_under_attack(params):
    global DEBUG
    rank, algo, env, start_seed, params = params
    n_trials = params.n_eval_episodes

    # Set seed
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    # import pdb; pdb.set_trace()

    attack_type = params.attack_type
    attack_epsilon = params.attack_epsilon
    attack_stepsize = attack_epsilon / params.attack_iteration
    if rank == 0:
        print("[INFO] Using %s attack: eps=%f, n_iters=%d, sz=%f" %
              (params.attack_type.upper(), attack_epsilon, params.attack_iteration, attack_stepsize))

    def perturb(state, type, attack_epsilon=None, attack_iteration=None, attack_stepsize=None,
               optimizer='pgd'):
        """" NOTE: This state is taken directly from environment, so it is un-normalized, when we
        return the perturbed state, it must be un-normalized
        """""
        state_tensor = tensor(state, algo._impl.device)
        # import pdb;pdb.set_trace()
        # Important: inside the attack functions, the state is assumed already normalized
        state_tensor = algo.scaler.transform(state_tensor)

        if type in ['random']:
            perturb_state = random_attack(
                state_tensor, attack_epsilon,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm,
            )

        elif type in ['critic_normal']:

            perturb_state = critic_normal_attack(
                state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm,
                optimizer=optimizer
            )

        elif type in ['actor_mad']:
            perturb_state = actor_mad_attack(
                state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min_norm, algo._impl._obs_max_norm,
                optimizer=optimizer
            )

        else:
            raise NotImplementedError
        assert np.linalg.norm(perturb_state.cpu()-state_tensor.cpu(),np.inf,1) < attack_epsilon + 1e-5, f"Over-epsilon {np.linalg.norm(perturb_state.cpu()-state_tensor.cpu(),np.inf,1) } "
        
        # if np.linalg.norm(perturb_state.cpu()-state_tensor.cpu(),np.inf,1) > attack_epsilon + 1e-5:
        #     import pdb;pdb.set_trace()
        #     print("ABC")
        
        # De-normalize state for return in original scale
        perturb_state = algo.scaler.reverse_transform(perturb_state)
        
        return perturb_state.squeeze().cpu().numpy()

    episode_rewards = []

    if DEBUG:
        act_diff_norm_inf = []
        act_diff_norm_2 = []
        state_diff_norm_inf = []
        state_diff_norm_2 = []
    for i in tqdm(range(n_trials), disable=(rank != 0), desc="{} attack".format(attack_type.upper())):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        state = env.reset()

        episode_reward = 0.0

        if DEBUG:
            act_diff_inf = []
            act_diff_2 = []
            state_diff_inf = []
            state_diff_2 = []


        while True:
            # take action
            state = make_sure_type_is_float32(state)
            
            if DEBUG:
                origin_state = state.copy()
                gt_action = algo.predict([state])[0]
                


            state = perturb(
                state,
                attack_type, attack_epsilon, params.attack_iteration, attack_stepsize,
                optimizer=params.optimizer
            )
            action = algo.predict([state])[0]
            if DEBUG:
                act_diff_inf.append(np.linalg.norm(action-gt_action,np.inf))
                act_diff_2.append(np.linalg.norm(action-gt_action))
                state_diff_inf.append(np.linalg.norm(state-origin_state,np.inf))
                state_diff_2.append(np.linalg.norm(state-origin_state))

            state, reward, done, _ = env.step(action)

            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)
        if DEBUG:
            act_diff_norm_inf.append(np.mean(np.array(act_diff_inf)))
            act_diff_norm_2.append(np.mean(np.array(act_diff_2)))
            state_diff_norm_inf.append(np.mean(np.array(state_diff_inf)))
            state_diff_norm_2.append(np.mean(np.array(state_diff_2)))
    if DEBUG:
        with open(f"Eval_debug_{attack_type}.txt") as f:
            f.write("Active:"+ ",".join([str(i) for i in episode_rewards])+"\n")
            f.write("Act_diff_inf:"+ ",".join([str(i) for i in act_diff_norm_inf])+"\n")
            f.write("Act_diff_2:"+ ",".join([str(i) for i in act_diff_norm_2])+"\n")
            f.write("state_diff_inf:"+ ",".join([str(i) for i in state_diff_norm_inf])+"\n")
            f.write("state_diff_2:"+ ",".join([str(i) for i in state_diff_norm_2])+"\n")
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
