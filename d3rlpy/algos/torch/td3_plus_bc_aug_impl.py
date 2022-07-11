# pylint: disable=too-many-ancestors

from typing import Optional, Sequence, Tuple
import copy

import torch
import torch.nn.functional as F
import numpy as np

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .td3_impl import TD3Impl
from ...adversarial_training import (
    clamp,
    ENV_OBS_RANGE,
    LinearSchedule,
)
from ...adversarial_training.attackers import (
    random_attack,
    actor_mad_attack,
    critic_normal_attack,
    critic_mqd_attack,
)


class TD3PlusBCAugImpl(TD3Impl):

    _alpha: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        transform: str = 'gaussian',
        transform_params: dict = None,
        env_name: str = '',
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

        self._transform = transform
        self._transform_params = transform_params

        if self._transform_params['epsilon_scheduler']['enable']:
            start_val = self._transform_params['epsilon_scheduler']['start']
            end_val = self._transform_params['epsilon_scheduler']['end']
            n_steps = self._transform_params['epsilon_scheduler']['steps']
            start_step = self._transform_params['epsilon_scheduler']['start_step']
            self.scheduler = LinearSchedule(start_val=start_val, end_val=end_val,
                                            n_steps=n_steps, start_step=start_step)
        else:
            self.scheduler = None

        env_name_ = env_name.split('-')
        self.env_name = env_name_[0] + '-' + env_name_[-1]
        self._obs_max = torch.Tensor(ENV_OBS_RANGE[self.env_name]['max']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))
        self._obs_min = torch.Tensor(ENV_OBS_RANGE[self.env_name]['min']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))

        self._obs_max_norm = self._obs_min_norm = None

    def init_range_of_norm_obs(self):
        self._obs_max_norm = self.scaler.transform(
            torch.Tensor(ENV_OBS_RANGE[self.env_name]['max']).to('cuda:{}'.format(
                self._use_gpu.get_id()))
        )
        self._obs_min_norm = self.scaler.transform(
            torch.Tensor(ENV_OBS_RANGE[self.env_name]['min']).to('cuda:{}'.format(
                self._use_gpu.get_id()))
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        actor_loss = lam * -q_t.mean()
        bc_loss = ((batch.actions - action) ** 2).mean()
        total_loss = actor_loss + bc_loss
        return total_loss, actor_loss, bc_loss

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_policy is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) \
        -> Tuple[np.ndarray, Tuple]:
        assert self._critic_optim is not None
        robust_type = self._transform_params.get('robust_type', None)
        critic_reg_coef = self._transform_params.get('critic_reg_coef', 0)

        # Override the value of epsilon by using scheduler
        epsilon = self.scheduler() if self.scheduler is not None else None

        if 'critic_drq' in robust_type:
            batch, batch_aug = self.do_augmentation(batch, for_critic=True)

            with torch.no_grad():
                q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
                q1_pred = q_prediction[0].cpu().detach()
                q2_pred = q_prediction[1].cpu().detach()

                q_prediction_adv = self._q_func(batch_aug.observations, batch_aug.actions, reduction="none")
                q1_pred_adv_diff = (q_prediction_adv[0].cpu().detach() - q1_pred).numpy().mean()
                q2_pred_adv_diff = (q_prediction_adv[1].cpu().detach() - q2_pred).numpy().mean()
                q1_pred = q1_pred.numpy().mean()
                q2_pred = q2_pred.numpy().mean()

            self._critic_optim.zero_grad()

            q_tpn = self.compute_target(batch)          # Compute target for clean data
            q_aug_tpn = self.compute_target(batch_aug)  # Compute target for augmented data
            q_tpn = (q_tpn + q_aug_tpn) / 2

            loss = (self.compute_critic_loss(batch, q_tpn) +
                    self.compute_critic_loss(batch_aug, q_tpn)) / 2

            loss.backward()
            self._critic_optim.step()

            extra_logs = (q_tpn.cpu().detach().numpy().mean(), q1_pred, q2_pred,
                          q1_pred_adv_diff, q2_pred_adv_diff)
        elif 'critic_reg' in robust_type:

            batch, batch_aug = self.do_augmentation(batch, for_critic=True, epsilon=epsilon)

            with torch.no_grad():
                # This is for logging
                q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
                q1_pred = q_prediction[0].cpu().detach().numpy().mean()
                q2_pred = q_prediction[1].cpu().detach().numpy().mean()

            with torch.no_grad():
                current_action = self._policy(batch.observations)
                gt_qval = self._q_func(batch.observations, current_action, "none").detach()

                current_action_adv = self._policy(batch_aug.observations).detach()


            self._critic_optim.zero_grad()

            q_tpn = self.compute_target(batch)  # Compute target for clean data

            loss = self.compute_critic_loss(batch, q_tpn)

            # Compute regularization
            qval_adv = self._q_func(batch.observations, current_action_adv, "none")
            q1_reg_loss = F.mse_loss(qval_adv[0], gt_qval[0])
            q2_reg_loss = F.mse_loss(qval_adv[1], gt_qval[1])
            critic_reg_loss = (q1_reg_loss + q2_reg_loss) / 2

            loss += critic_reg_coef * critic_reg_loss

            loss.backward()
            self._critic_optim.step()

            # Compute the difference w.r.t. the current policy applied on states
            q1_pred_adv_diff = (qval_adv[0] - gt_qval[0]).detach().cpu().numpy().mean()
            q2_pred_adv_diff = (qval_adv[1] - gt_qval[1]).detach().cpu().numpy().mean()

            extra_logs = (q_tpn.cpu().detach().numpy().mean(), q1_pred, q2_pred,
                          q1_pred_adv_diff, q2_pred_adv_diff, critic_reg_coef * critic_reg_loss.item(),
                          epsilon)

        else:
            q1_pred_adv_diff, q2_pred_adv_diff = 0.0, 0.0
            with torch.no_grad():
                q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
                q1_pred = q_prediction[0].cpu().detach().numpy().mean()
                q2_pred = q_prediction[1].cpu().detach().numpy().mean()

            self._critic_optim.zero_grad()

            q_tpn = self.compute_target(batch)

            loss = self.compute_critic_loss(batch, q_tpn)

            loss.backward()
            self._critic_optim.step()

            extra_logs = (q_tpn.cpu().detach().numpy().mean(), q1_pred, q2_pred,
                          q1_pred_adv_diff, q2_pred_adv_diff)

        return loss.cpu().detach().numpy(), extra_logs

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) \
        -> Tuple[np.ndarray, Tuple]:
        assert self._q_func is not None
        assert self._actor_optim is not None
        robust_type = self._transform_params.get('robust_type', None)

        if 'actor_mad' in robust_type:
            actor_reg_coef = self._transform_params.get('actor_reg_coef', 0)

            batch, batch_aug = self.do_augmentation(batch, for_critic=False)

            # Q function should be inference mode for stability
            self._q_func.eval()

            self._actor_optim.zero_grad()

            loss, actor_loss, bc_loss = self.compute_actor_loss(batch)

            if actor_reg_coef > 0:
                action_reg_loss =  ((self._policy(batch_aug.observations) - batch.actions) ** 2).mean()
                loss += actor_reg_coef * action_reg_loss
            else:
                action_reg_loss = 0

            loss.backward()
            self._actor_optim.step()

            action_reg_loss = action_reg_loss.item() if action_reg_loss > 0 else 0
            extra_logs = (actor_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy(),
                          action_reg_loss)

        elif 'actor_on_adv' in robust_type:
            actor_reg_coef = self._transform_params.get('actor_reg_coef', 0)
            prob_of_actor_on_adv = self._transform_params.get('prob_of_actor_on_adv', 0)
            assert actor_reg_coef == 0 and (0 < prob_of_actor_on_adv <= 1)

            batch, batch_aug = self.do_augmentation(batch, for_critic=False)

            # Q function should be inference mode for stability
            self._q_func.eval()

            self._actor_optim.zero_grad()

            prob = np.random.uniform(1)
            if prob <= prob_of_actor_on_adv:
                loss, actor_loss, bc_loss = self.compute_actor_loss(batch_aug)
            else:
                loss, actor_loss, bc_loss = self.compute_actor_loss(batch)

            action_reg_loss = 0

            loss.backward()
            self._actor_optim.step()

            extra_logs = (actor_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy(),
                          action_reg_loss)

        else:

            # Q function should be inference mode for stability
            self._q_func.eval()

            self._actor_optim.zero_grad()

            loss, actor_loss, bc_loss = self.compute_actor_loss(batch)

            loss.backward()
            self._actor_optim.step()

            extra_logs = (actor_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy())

        return loss.cpu().detach().numpy(), extra_logs

    def do_augmentation(self, batch: TorchMiniBatch, for_critic=True, epsilon=None):
        """" NOTE: Assume obs, next_obs are already normalized """""

        batch_aug = copy.deepcopy(batch)
        # Transforming the copied batch

        assert self._transform in ['adversarial_training']
        #### Using PGD with Linf-norm
        epsilon = self._transform_params.get('epsilon', None) if epsilon is None else epsilon
        num_steps = self._transform_params.get('num_steps', None)
        step_size = self._transform_params.get('step_size', None)
        attack_type = self._transform_params.get('attack_type', None)
        attack_type_for_actor = self._transform_params.get('attack_type_for_actor', None)
        optimizer = self._transform_params.get('optimizer', 'pgd')

        if (attack_type_for_actor is not None) and (for_critic is False):
            # This attack is specified for attack actor
            attack_type = attack_type_for_actor

        assert (epsilon is not None) and (num_steps is not None) and \
               (step_size is not None) and (attack_type is not None)

        if attack_type in ['random']:
            adv_x = random_attack(batch_aug._observations, epsilon,
                                  self._obs_min_norm, self._obs_max_norm)
            batch_aug._observations = adv_x

            adv_x = random_attack(batch_aug._next_observations, epsilon,
                                  self._obs_min_norm, self._obs_max_norm)
            batch_aug._next_observations = adv_x

        elif attack_type in ['critic_normal']:
            adv_x = critic_normal_attack(batch_aug._observations,
                                         self._policy, self._q_func,
                                         epsilon, num_steps, step_size,
                                         self._obs_min_norm, self._obs_max_norm,
                                         optimizer=optimizer)
            batch_aug._observations = adv_x

            # adv_x = critic_normal_attack(batch_aug._next_observations,
            #                              self._policy, self._q_func,
            #                              epsilon, num_steps, step_size,
            #                              self._obs_min_norm, self._obs_max_norm,
            #                              optimizer=optimizer)
            # batch_aug._next_observations = adv_x

        elif attack_type in ['critic_mqd']:
            adv_x = critic_mqd_attack(batch_aug._observations,
                                      batch_aug._actions,
                                      self._policy, self._q_func,
                                      epsilon, num_steps, step_size,
                                      self._obs_min_norm, self._obs_max_norm)
            batch_aug._observations = adv_x


        elif attack_type in ['actor_mad']:
            adv_x = actor_mad_attack(batch_aug._observations,
                                     self._policy, self._q_func,
                                     epsilon, num_steps, step_size,
                                     self._obs_min_norm, self._obs_max_norm,
                                     optimizer=optimizer)
            batch_aug._observations = adv_x

            adv_x = actor_mad_attack(batch_aug._next_observations,
                                     self._policy, self._q_func,
                                     epsilon, num_steps, step_size,
                                     self._obs_min_norm, self._obs_max_norm,
                                     optimizer=optimizer)
            batch_aug._next_observations = adv_x

        else:
            raise NotImplementedError

        return batch, batch_aug

