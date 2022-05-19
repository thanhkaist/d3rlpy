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
    parser.add_argument('--n_steps', type=int, default=500000)
    parser.add_argument('--n_eval_episodes', type=int, default=5)

    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    td3 = d3rlpy.algos.TD3PlusBC(actor_learning_rate=3e-4,
                                 critic_learning_rate=3e-4,
                                 batch_size=256,
                                 target_smoothing_sigma=0.2,
                                 target_smoothing_clip=0.5,
                                 alpha=2.5,
                                 update_actor_interval=2,
                                 scaler="standard",
                                 use_gpu=args.gpu)

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
            'td_error': d3rlpy.metrics.td_error_scorer,
            'value_estimation_std': d3rlpy.metrics.value_estimation_std_scorer,
            'initial_state_value_estimation': d3rlpy.metrics.initial_state_value_estimation_scorer
        },
        wandb_project=args.project,
        use_wandb=args.wandb,
        experiment_name=f"TD3_BC_{args.dataset}_{args.exp}"
    )


if __name__ == '__main__':
    main()
