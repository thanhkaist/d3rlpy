from typing import List, Optional, Union, cast

import torch
from torch import nn

from .base import ContinuousQFunction, DiscreteQFunction

import torch.nn.functional as F
import numpy as np
from .utility import compute_reduce

# Library for computing output bound of network
from auto_LiRPA.bound_ops import BoundParams
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    # reduction beased on expectation
    mean = y.mean(dim=-1)
    if reduction == "min":
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


class EnsembleQFunction(nn.Module):  # type: ignore
    _action_size: int
    _q_funcs: nn.ModuleList

    def __init__(
        self,
        q_funcs: Union[List[DiscreteQFunction], List[ContinuousQFunction]],
    ):
        super().__init__()
        self._action_size = q_funcs[0].action_size
        self._q_funcs = nn.ModuleList(q_funcs)

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        assert target.ndim == 2

        td_sum = torch.tensor(
            0.0, dtype=torch.float32, device=observations.device
        )
        for q_func in self._q_funcs:
            loss = q_func.compute_error(
                observations=observations,
                actions=actions,
                rewards=rewards,
                target=target,
                terminals=terminals,
                gamma=gamma,
                reduction="none",
            )
            td_sum += loss.mean()
        return td_sum

    def _compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        values_list: List[torch.Tensor] = []
        for q_func in self._q_funcs:
            target = q_func.compute_target(x, action)
            values_list.append(target.reshape(1, x.shape[0], -1))

        values = torch.cat(values_list, dim=0)

        if action is None:
            # mean Q function
            if values.shape[2] == self._action_size:
                return _reduce_ensemble(values, reduction)
            # distributional Q function
            n_q_funcs = values.shape[0]
            values = values.view(n_q_funcs, x.shape[0], self._action_size, -1)
            return _reduce_quantile_ensemble(values, reduction)

        if values.shape[2] == 1:
            return _reduce_ensemble(values, reduction, lam=lam)

        return _reduce_quantile_ensemble(values, reduction, lam=lam)

    @property
    def q_funcs(self) -> nn.ModuleList:
        return self._q_funcs


class EnsembleDiscreteQFunction(EnsembleQFunction):
    def forward(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x).view(1, x.shape[0], self._action_size))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, reduction, lam)


class EnsembleContinuousQFunction(EnsembleQFunction):
    def forward(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, reduction, lam)


class WrapperBoundEnsembleContinuousQFunction(nn.Module):
    _action_size: int
    _q_funcs: nn.ModuleList

    def __init__(self, q_func, observation_shape, action_shape, device, use_full_backward=False):
        super().__init__()
        self.use_full_backward = use_full_backward

        self.unwrapped_q = q_func   # This is original unwrapped object
        self._action_size = q_func.q_funcs[0].action_size

        for i in range(len(q_func.q_funcs)):
            self.unwrapped_q._q_funcs[i] = BoundedModule(
                model=q_func.q_funcs[i],
                global_input=(torch.empty(size=(1, observation_shape)), torch.empty(size=(1, action_shape))),
                device=device
            )

    def reset_weight(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, BoundParams):
                params = m.forward_value
                if params.ndim == 2:
                    torch.nn.init.kaiming_uniform_(params, a=np.sqrt(5))
                else:
                    torch.nn.init.normal_(params)

        self.unwrapped_q.apply(weight_reset)

    def forward(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for q_func in self.unwrapped_q._q_funcs:
            q = q_func(x, action, method_opt="forward")
            values.append(q.view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, reduction))

    def compute_target(self, x: torch.Tensor, action: torch.Tensor,
                       reduction: str = "min", lam: float = 0.75, ) -> torch.Tensor:
        # Define compute target again

        return self._compute_target(x, action, reduction, lam)

    def _compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor] = None,
        reduction: str = "min", lam: float = 0.75,) -> torch.Tensor:
        values_list: List[torch.Tensor] = []
        for q_func in self.unwrapped_q._q_funcs:
            target = self.compute_target_for_single_q(q_func, x, action)
            values_list.append(target.reshape(1, x.shape[0], -1))

        values = torch.cat(values_list, dim=0)

        if action is None:
            # mean Q function
            if values.shape[2] == self._action_size:
                return _reduce_ensemble(values, reduction)
            # distributional Q function
            n_q_funcs = values.shape[0]
            values = values.view(n_q_funcs, x.shape[0], self._action_size, -1)
            return _reduce_quantile_ensemble(values, reduction)

        if values.shape[2] == 1:
            return _reduce_ensemble(values, reduction, lam=lam)

        return _reduce_quantile_ensemble(values, reduction, lam=lam)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor,
        rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float = 0.99,
    ) -> torch.Tensor:
        assert target.ndim == 2

        td_sum = torch.tensor(
            0.0, dtype=torch.float32, device=observations.device
        )
        for q_func in self.unwrapped_q._q_funcs:
            loss = self.compute_error_for_single_q(
                q_func,
                observations=observations,
                actions=actions,
                rewards=rewards,
                target=target,
                terminals=terminals,
                gamma=gamma,
                reduction="none",
            )
            td_sum += loss.mean()
        return td_sum

    def compute_target_for_single_q(self, q, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return q(x, action, method_opt="forward")

    def compute_error_for_single_q(self, q, observations: torch.Tensor, actions: torch.Tensor,
                                   rewards: torch.Tensor, target: torch.Tensor,
                                   terminals: torch.Tensor, gamma: float = 0.99,
                                   reduction: str = "mean", ):

        value = q(observations, actions, method_opt="forward")
        y = rewards + gamma * target * (1 - terminals)
        loss = F.mse_loss(value, y, reduction="none")
        return compute_reduce(loss, reduction)

    @property
    def q_funcs(self) -> nn.ModuleList:
        return self.unwrapped_q._q_funcs

    def load_state_dict(self, state_dict, strict=False):
        self.unwrapped_q.load_state_dict(state_dict, strict)

    # Obtain element-wise lower and upper bounds for actor network through convex relaxations.
    def compute_bound(self, x_lb, x_ub, a_lb, a_ub, x=None, a=None, beta=0, eps=None, norm=np.inf,
                      q_idx=None):

        values_lb, values_ub = [], []
        x_perturb = PerturbationLpNorm(norm=norm, eps=eps, x_L=x_lb, x_U=x_ub)
        x = BoundedTensor(x, x_perturb)
        a_perturb = PerturbationLpNorm(norm=norm, eps=eps, x_L=a_lb, x_U=a_ub)
        a = BoundedTensor(a, a_perturb)

        if q_idx is None:
            # Return upper bound & lower bound for both q functions
            for q_func in self.unwrapped_q._q_funcs:
                if self.use_full_backward:
                    q_lb_bw, q_ub_bw = q_func.compute_bounds(x=(x, a), IBP=False, method="backward")
                    q_lb, q_ub = q_lb_bw, q_ub_bw

                else:
                    q_lb_ibp, q_ub_ibp = q_func.compute_bounds(x=(x, a), IBP=True, method=None)
                    if beta > 1e-10:
                        q_lb_bw, q_ub_bw = q_func.compute_bounds(IBP=False, method="backward")
                        q_lb = q_lb_bw * beta + q_lb_ibp * (1.0 - beta)
                        q_ub = q_ub_bw * beta + q_ub_ibp * (1.0 - beta)
                    else:
                        q_lb, q_ub = q_lb_ibp, q_ub_ibp

                values_lb.append(q_lb)
                values_ub.append(q_ub)

        else:
            assert q_idx in [0, 1], "Index of Q function used among two Q(s), applied to TD3"
            if self.use_full_backward:
                q_lb_bw, q_ub_bw = self.unwrapped_q._q_funcs[q_idx].compute_bounds(
                    x=(x, a), IBP=False, method="backward")
                q_lb, q_ub = q_lb_bw, q_ub_bw

            else:
                q_lb_ibp, q_ub_ibp = self.unwrapped_q._q_funcs[q_idx].compute_bounds(
                    x=(x, a), IBP=True, method=None)
                if beta > 1e-10:
                    q_lb_bw, q_ub_bw = self.unwrapped_q._q_funcs[q_idx].compute_bounds(
                        IBP=False, method="backward")
                    q_lb = q_lb_bw * beta + q_lb_ibp * (1.0 - beta)
                    q_ub = q_ub_bw * beta + q_ub_ibp * (1.0 - beta)
                else:
                    q_lb, q_ub = q_lb_ibp, q_ub_ibp

            values_lb.append(q_lb)
            values_ub.append(q_ub)

        return values_lb, values_ub
