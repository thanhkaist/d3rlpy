import argparse
import d3rlpy
from sklearn.model_selection import train_test_split
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


def set_name_wandb_project(dataset):
    project_name = None
    if 'hopper' in dataset:
        project_name = 'HOPPER-J'
    elif 'walker' in dataset:
        project_name = 'WALKER-J'
    elif 'halfcheetah' in dataset:
        project_name = 'CHEETAH-J'

    return project_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
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

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    td3 = d3rlpy.algos.TD3PlusBC(actor_learning_rate=3e-4,
                                 critic_learning_rate=3e-4,
                                 batch_size=256,
                                 target_smoothing_sigma=args.target_smoothing_sigma,
                                 target_smoothing_clip=args.target_smoothing_clip,
                                 alpha=2.5,
                                 update_actor_interval=2,
                                 scaler="standard",
                                 use_gpu=args.gpu,
                                 env_name=args.dataset,
                                 )

    scorer_funcs = {
        'environment': d3rlpy.metrics.evaluate_on_environment(env, n_trials=args.n_eval_episodes),
        'critic_normal_environment': d3rlpy.metrics.evaluate_on_environment_with_attack(
            env,
            n_trials=args.n_eval_episodes,
            attack_type="critic_normal",
            attack_epsilon=args.epsilon,
            attack_iteration=args.num_steps,
            attack_stepsize=args.epsilon / args.num_steps
        ),
        'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
        'td_error': d3rlpy.metrics.td_error_scorer,
        'value_estimation_std': d3rlpy.metrics.value_estimation_std_scorer,
        'initial_state_value_estimation': d3rlpy.metrics.initial_state_value_estimation_scorer
    }

    td3.fit(
        dataset.episodes,
        eval_episodes=test_episodes,
        n_steps=args.n_steps,
        n_steps_per_epoch=1000,
        save_interval=10,
        logdir=args.logdir,
        scorers=scorer_funcs,
        wandb_project=set_name_wandb_project(args.dataset),
        use_wandb=args.wandb,
        experiment_name=f"TD3_BC_{ENV_NAME_MAPPING[args.dataset]}_{args.exp}"
    )


if __name__ == '__main__':
    main()
