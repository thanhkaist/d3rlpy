import argparse
import d3rlpy
from sklearn.model_selection import train_test_split


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
    parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--project', type=str, default='WALKER')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--logdir', type=str, default='d3rlpy_logs')
    parser.add_argument('--n_steps', type=int, default=500000)
    parser.add_argument('--n_eval_episodes', type=int, default=5)

    parser.add_argument('--noise_test', type=str, default='uniform')
    parser.add_argument('--noise_test_eps', type=float, default=1e-4)

    SUPPORTED_TRANSFORMS = ['gaussian', 'adversarial_training']
    parser.add_argument('--transform', type=str, default='gaussian', choices=SUPPORTED_TRANSFORMS)
    parser.add_argument('--epsilon', type=float, default=3e-4)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--step_size', type=float, default=2.5e-5)
    parser.add_argument('--norm_min_max', action='store_true')
    parser.add_argument('--adv_version', type=str, default='a1_d1')

    parser.add_argument('--use_action_from_data', action='store_true')
    parser.add_argument('--act_on_adv', action='store_true')

    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    # Define version:
    """
    - a1_d1: (1) Generating adv example by using actor loss with constant action. (2) Robust training
    similar to DrQ.
    - a2_d1: (1) Generating adv example by using critic loss (Bellman error). (2) Robust training
    similar to DrQ. (Cannot use)
    - a1_d2: (1) Generating adv example by using actor loss with constant action. (2) Robust training
    similar to DrQ, but the target is computed for clean data, not adversarial data.
    - a2_d2: (1) Generating adv example by using critic loss (Bellman error). (2) Robust training
    similar to DrQ, but the target is computed for clean data, not adversarial data.
    - a3_d2: (1) Generating adv example (both state and action) by using actor loss with constant
    action. (2) Robust training similar to DrQ, but the target is computed for clean data, not adversarial data.
    -
    """
    transform_params = dict(
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        step_size=args.step_size,
        norm_min_max=args.norm_min_max,
        adv_version=args.adv_version,
        use_action_from_data=args.use_action_from_data,
        act_on_adv=args.act_on_adv
    )
    td3 = d3rlpy.algos.TD3PlusBCAug(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        target_smoothing_sigma=0.2,
        target_smoothing_clip=0.5,
        alpha=2.5,
        update_actor_interval=2,
        scaler="standard",
        use_gpu=args.gpu,
        transform=args.transform,
        transform_params=transform_params,
        env_name=args.dataset,
    )

    td3.fit(
        dataset.episodes,
        eval_episodes=test_episodes,
        n_steps=args.n_steps,
        n_steps_per_epoch=1000,
        save_interval=10,
        logdir=args.logdir,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env, n_trials=args.n_eval_episodes),
            'noise_environment': d3rlpy.metrics.evaluate_on_noise_environment(env,
                                                                              n_trials=args.n_eval_episodes,
                                                                              noise_type=args.noise_test,
                                                                              eps_noise=args.noise_test_eps),
            'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            'td_error': d3rlpy.metrics.td_error_scorer,
            'value_estimation_std': d3rlpy.metrics.value_estimation_std_scorer,
            'initial_state_value_estimation': d3rlpy.metrics.initial_state_value_estimation_scorer
        },
        wandb_project=args.project,
        use_wandb=args.wandb,
        experiment_name=f"TD3_BC_{ENV_NAME_MAPPING[args.dataset]}_{args.exp}"
    )


if __name__ == '__main__':
    main()
