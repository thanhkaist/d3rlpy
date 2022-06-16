import argparse
import gym

import d3rlpy
from d3rlpy.online.explorers import NormalNoise
from d3rlpy.preprocessing.scalers import StandardScaler


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
    parser.add_argument('--dataset', type=str, default='walker2d-expert-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--project', type=str, default='WALKER')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--logdir', type=str, default='d3rlpy_logs')
    parser.add_argument('--n_steps', type=int, default=500000)
    parser.add_argument('--n_eval_episodes', type=int, default=5)

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

    if 'walker' in args.dataset:
        dataset1, _ = d3rlpy.datasets.get_dataset('walker2d-random-v0')
        dataset2, _ = d3rlpy.datasets.get_dataset('walker2d-medium-v0')
        dataset3, _ = d3rlpy.datasets.get_dataset('walker2d-medium-replay-v0')
        dataset4, _ = d3rlpy.datasets.get_dataset('walker2d-expert-v0')

        dataset1.extend(dataset2)
        dataset1.extend(dataset3)
        dataset1.extend(dataset4)

        scaler = StandardScaler(dataset1)
    elif 'hopper' in args.dataset:
        raise NotImplementedError
    elif 'halfcheetah' in args.dataset:
        raise NotImplementedError
    else:
        raise NotImplementedError

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
        n_steps=2000000,
        n_steps_per_epoch=2000,
        update_start_step=25000,
        save_interval=5,
        explorer=NormalNoise(mean=0.0, std=0.1),
        logdir=args.logdir,
        wandb_project=args.project,
        use_wandb=args.wandb,
        experiment_name=f"{ENV_NAME_MAPPING[args.dataset]}_{args.exp}",
    )


if __name__ == '__main__':
    main()
