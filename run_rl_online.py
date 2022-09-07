import argparse
import gym

import d3rlpy
from d3rlpy.online.explorers import NormalNoise
from d3rlpy.preprocessing.scalers import StandardScaler
from d3rlpy.adversarial_training.utility import set_name_wandb_project


ENV_NAME_MAPPING = {
    'walker2d-random-v0': 'w2d-r',
    'walker2d-medium-v0': 'w2d-m',
    'walker2d-medium-replay-v0': 'w2d-m-re',
    'walker2d-medium-expert-v0': 'w2d-m-e',
    'walker2d-expert-v0': 'w2d-e',
    'hopper-random-v0': 'hop-r',
    'hopper-medium-v0': 'hop-m',
    'hopper-medium-replay-v0': 'hop-m-re',
    'hopper-medium-expert-v0': 'hop-m-e',
    'hopper-expert-v0': 'hop-e',
    'halfcheetah-random-v0': 'che-r',
    'halfcheetah-medium-v0': 'che-m',
    'halfcheetah-medium-replay-v0': 'che-m-re',
    'halfcheetah-medium-expert-v0': 'che-m-e',
    'halfcheetah-expert-v0': 'che-e'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='TD3', choices=['TD3', 'SAC'])
    parser.add_argument('--dataset', type=str, default='walker2d-expert-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--project', type=str, default='WALKER')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--logdir', type=str, default='d3rlpy_logs')
    parser.add_argument('--n_steps', type=int, default=2000000)
    parser.add_argument('--eval_interval', type=int, default=25)
    parser.add_argument('--n_eval_episodes', type=int, default=10)

    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--step_size', type=float, default=0.01)

    # For analyzing
    parser.add_argument('--target_smoothing_sigma', type=float, default=0.2)
    parser.add_argument('--target_smoothing_clip', type=float, default=0.5)

    args = parser.parse_args()

    env = gym.make(args.dataset)
    eval_env = gym.make(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    eval_env.seed(args.seed)

    if args.algo == 'TD3':
        if 'walker' in args.dataset:
            dataset1, _ = d3rlpy.datasets.get_dataset('walker2d-random-v0')
            dataset2, _ = d3rlpy.datasets.get_dataset('walker2d-medium-v0')
            dataset3, _ = d3rlpy.datasets.get_dataset('walker2d-medium-replay-v0')
            dataset4, _ = d3rlpy.datasets.get_dataset('walker2d-expert-v0')

            dataset1.extend(dataset2)
            dataset1.extend(dataset3)
            dataset1.extend(dataset4)

            scaler = StandardScaler(dataset1)
        elif 'hopper' in args.dataset and args.algo == 'TD3':
            dataset1, _ = d3rlpy.datasets.get_dataset('hopper-random-v0')
            dataset2, _ = d3rlpy.datasets.get_dataset('hopper-medium-v0')
            dataset3, _ = d3rlpy.datasets.get_dataset('hopper-medium-replay-v0')
            dataset4, _ = d3rlpy.datasets.get_dataset('hopper-expert-v0')

            dataset1.extend(dataset2)
            dataset1.extend(dataset3)
            dataset1.extend(dataset4)

            scaler = StandardScaler(dataset1)
        elif 'halfcheetah' in args.dataset and args.algo == 'TD3':
            dataset1, _ = d3rlpy.datasets.get_dataset('halfcheetah-random-v0')
            dataset2, _ = d3rlpy.datasets.get_dataset('halfcheetah-medium-v0')
            dataset3, _ = d3rlpy.datasets.get_dataset('halfcheetah-medium-replay-v0')
            dataset4, _ = d3rlpy.datasets.get_dataset('halfcheetah-expert-v0')

            dataset1.extend(dataset2)
            dataset1.extend(dataset3)
            dataset1.extend(dataset4)

            scaler = StandardScaler(dataset1)
        else:
            raise NotImplementedError

    if args.algo == 'TD3':
        # setup algorithm
        td3 = d3rlpy.algos.TD3(
            batch_size=256,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            target_smoothing_sigma=0.2,
            target_smoothing_clip=0.5,
            use_gpu=args.gpu,
            scaler=scaler,
        )

        # prepare replay buffer
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

        # start training
        td3.fit_online(
            env,
            buffer,
            eval_env=eval_env,
            n_steps=args.n_steps,
            n_steps_per_epoch=2000,
            update_start_step=25000,
            save_interval=10,
            explorer=NormalNoise(mean=0.0, std=0.1),
            logdir=args.logdir,
            wandb_project=set_name_wandb_project(args.dataset),
            use_wandb=args.wandb,
            experiment_name=f"{ENV_NAME_MAPPING[args.dataset]}_{args.exp}",
            eval_interval=args.eval_interval,
        )
    elif args.algo == 'SAC':
        sac = d3rlpy.algos.SAC(
            batch_size=256,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            temp_learning_rate=3e-4,
            use_gpu=args.gpu
        )

        # replay buffer for experience replay
        buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

        # start training
        sac.fit_online(
            env,
            buffer,
            eval_env=eval_env,
            n_steps=1000000,
            n_steps_per_epoch=10000,
            update_interval=1,
            update_start_step=1000,
            save_interval=10,
            logdir=args.logdir,
            wandb_project=set_name_wandb_project(args.dataset),
            use_wandb=args.wandb,
            experiment_name=f"{ENV_NAME_MAPPING[args.dataset]}_{args.exp}",
            eval_interval=args.eval_interval,
        )


if __name__ == '__main__':
    main()
