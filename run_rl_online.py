import os
import argparse
import gym
import glob
import h5py
import copy

import numpy as np

import d3rlpy
from d3rlpy.preprocessing.scalers import StandardScaler


def set_name_wandb_project(dataset):
    project_name = 'IKEA-TEST'

    return project_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='TD3', choices=['TD3', 'SAC'])
    parser.add_argument('--dataset', type=str, default='walker2d-medium-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--project', type=str, default='WALKER')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--entity', type=str, default='tunglm')
    parser.add_argument('--logdir', type=str, default='d3rlpy_logs')
    parser.add_argument('--n_steps', type=int, default=2000000)
    parser.add_argument('--n_steps_collect_data', type=int, default=10000000)
    parser.add_argument('--n_steps_per_epoch', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--n_eval_episodes', type=int, default=10)

    parser.add_argument('--no_replacement', action='store_true', default=False)
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--stats_update_interval', type=int, default=1000)

    parser.add_argument('--loss_type', type=str, default="normal", choices=["normal", "mad_loss"])
    parser.add_argument('--attack_type', type=str, default="actor_state_linf")
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--actor_reg', type=float, default=0.5)

    # Scope for Vector Quantization representation
    parser.add_argument('--use_vq_in', action='store_true', default=False)
    parser.add_argument('--codebook_update_type', type=str, default="ema", choices=["ema", "sgd"])
    parser.add_argument('--n_embeddings', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=1)
    parser.add_argument('--vq_decay', type=float, default=0.99)
    parser.add_argument('--vq_loss_weight', type=float, default=1.0)
    parser.add_argument('--autoscale_vq_loss', action='store_true', default=False)
    parser.add_argument('--scale_factor', type=float, default=60.0)
    parser.add_argument('--n_steps_allow_update_cb', type=int, default=10000000)
    parser.add_argument('--n_steps_start_at', type=int, default=0)

    parser.add_argument('--vq_decay_scheduler', action='store_true', default=False)
    parser.add_argument('--vq_decay_start_val', type=float, default=0.5)
    parser.add_argument('--vq_decay_end_val', type=float, default=0.99)
    parser.add_argument('--vq_decay_start_step', type=int, default=0)
    parser.add_argument('--vq_decay_end_step', type=int, default=1000000)

    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--ckpt', type=str, default='none')
    parser.add_argument('--ckpt_steps', type=str, default='model_500000.pt')
    parser.add_argument('--load_buffer', action='store_true', default=False)
    parser.add_argument('--backup_file', action='store_true')

    args = parser.parse_args()

    env = gym.make(args.dataset)
    eval_env = gym.make(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    if args.standardization:
        scaler = StandardScaler(mean=np.zeros(env.observation_space.shape), std=np.ones(env.observation_space.shape))
    else:
        scaler = None

    if args.algo == 'TD3':
        raise NotImplementedError
    elif args.algo == 'SAC':

        sac = d3rlpy.algos.SAC(
            batch_size=256,
            use_gpu=args.gpu,
            scaler=scaler,
            replacement=not args.no_replacement,
            env_name=args.dataset,
        )

        buffer_size = 1000000
        if args.n_steps < buffer_size:
            buffer_size = args.n_steps

        # replay buffer for experience replay
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=buffer_size, env=env,
                                                    compute_statistical=args.standardization)

        # start training
        sac.fit_online(
            env,
            buffer,
            eval_env=eval_env,
            n_steps=args.n_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            update_interval=1,
            update_start_step=1000,
            save_interval=args.save_interval,
            logdir=args.logdir,
            wandb_project=set_name_wandb_project(args.dataset),
            use_wandb=args.wandb,
            entity=args.entity,
            experiment_name=f"{args.exp}",
            eval_interval=args.eval_interval,
            standardization=args.standardization,
            stats_update_interval=args.stats_update_interval,
            backup_file=True,
        )


if __name__ == '__main__':
    main()
