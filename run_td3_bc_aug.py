import argparse
import d3rlpy
from sklearn.model_selection import train_test_split
from d3rlpy.argument_utility import check_scaler


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

    SUPPORTED_TRANSFORMS = ['gaussian', 'adversarial_training']
    parser.add_argument('--transform', type=str, default='gaussian', choices=SUPPORTED_TRANSFORMS)
    parser.add_argument('--epsilon', type=float, default=3e-4)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--step_size', type=float, default=2.5e-5)
    parser.add_argument('--norm_min_max', action='store_true')

    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    # TODO: Manually get scaler
    transitions = []
    for episode in dataset.episodes:
        transitions += episode.transitions
    scaler = check_scaler("standard")
    scaler.fit(transitions)

    transform_params = dict(
        epsilon=args.epsilon,
        num_steps=args.num_steps,
        step_size=args.step_size,
        norm_min_max=args.norm_min_max
    )
    td3 = d3rlpy.algos.TD3PlusBCAug(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        target_smoothing_sigma=0.2,
        target_smoothing_clip=0.5,
        alpha=2.5,
        update_actor_interval=2,
        scaler=None,
        use_gpu=args.gpu,
        transform=args.transform,
        transform_params=transform_params,
        env_name=args.dataset,
        custom_scaler=scaler
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
            'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
        },
        wandb_project=args.project,
        use_wandb=args.wandb,
        experiment_name=f"TD3_BC_{args.dataset}_{args.exp}"
    )


if __name__ == '__main__':
    main()
