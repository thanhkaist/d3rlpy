import argparse
import d3rlpy
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--project', type=str, default='WALKER')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--logdir', type=str, default='d3rlpy_logs')
    parser.add_argument('--n_steps', type=int, default=1000000)

    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--alpha_threshold', type=float, default=-1.0)
    parser.add_argument('--conservative_weight', type=float, default=5.0)

    SUPPORTED_TRANSFORMS = ['gaussian']
    parser.add_argument('--transform', type=str, default='gaussian', choices=SUPPORTED_TRANSFORMS)
    parser.add_argument('--epsilon', type=float, default=3e-4)
    parser.add_argument('--norm_min_max', action='store_true')
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    conservative_weight = args.conservative_weight
    alpha_threshold = args.alpha_threshold
    policy_eval_start = 40000

    transform_params = dict(
        epsilon=args.epsilon,
        norm_min_max=args.norm_min_max
    )
    cql = d3rlpy.algos.CQLAug(
        actor_learning_rate=args.policy_lr,
        critic_learning_rate=3e-4,
        temp_learning_rate=args.policy_lr,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=conservative_weight,
        alpha_threshold=alpha_threshold,
        use_gpu=args.gpu,
        policy_eval_start=policy_eval_start,
        transform=args.transform,
        transform_params=transform_params,
        env_name=args.dataset
    )

    cql.fit(
        dataset.episodes,
        eval_episodes=test_episodes,
        n_steps=args.n_steps,
        n_steps_per_epoch=1000,
        save_interval=10,
        logdir=args.logdir,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env, n_trials=5),
            'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
        },
        wandb_project=args.project,
        use_wandb=args.wandb,
        experiment_name=f"CQL_{args.dataset}_{args.exp}"
    )


if __name__ == '__main__':
    main()
