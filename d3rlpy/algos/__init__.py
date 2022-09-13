from typing import Any, Dict, Type

from .awac import AWAC
from .base import AlgoBase
from .bc import BC, DiscreteBC
from .bcq import BCQ, DiscreteBCQ
from .bear import BEAR
from .combo import COMBO
from .cql import CQL, DiscreteCQL
from .crr import CRR
from .ddpg import DDPG
from .ddpg_plus_bc import DDPGPlusBC
from .dqn import DQN, DoubleDQN
from .iql import IQL
from .mopo import MOPO
from .nfq import NFQ
from .plas import PLAS, PLASWithPerturbation
from .random_policy import DiscreteRandomPolicy, RandomPolicy
from .sac import SAC, DiscreteSAC
from .td3 import TD3
from .td3_plus_bc import TD3PlusBC
from .td3_relational import TD3PlusRelation
from .cql_aug import CQLAug, DiscreteCQLAug
from .td3_plus_bc_aug import TD3PlusBCAug

__all__ = [
    "AlgoBase",
    "AWAC",
    "BC",
    "DiscreteBC",
    "BCQ",
    "DiscreteBCQ",
    "BEAR",
    "COMBO",
    "CQL",
    "DiscreteCQL",
    "CQLAug",
    "DiscreteCQLAug",
    "CRR",
    "DDPG",
    "DDPGPlusBC",
    "DQN",
    "DoubleDQN",
    "IQL",
    "MOPO",
    "NFQ",
    "PLAS",
    "PLASWithPerturbation",
    "SAC",
    "DiscreteSAC",
    "TD3",
    "TD3PlusBC",
    "TD3PlusBCAug",
    "RandomPolicy",
    "DiscreteRandomPolicy",
    "get_algo",
    "create_algo",
]


DISCRETE_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "bc": DiscreteBC,
    "bcq": DiscreteBCQ,
    "cql": DiscreteCQL,
    "cql_aug": DiscreteCQLAug,
    "dqn": DQN,
    "double_dqn": DoubleDQN,
    "nfq": NFQ,
    "sac": DiscreteSAC,
    "random": DiscreteRandomPolicy,
}

CONTINUOUS_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "awac": AWAC,
    "bc": BC,
    "bcq": BCQ,
    "bear": BEAR,
    "combo": COMBO,
    "cql": CQL,
    "cql_aug": CQLAug,
    "crr": CRR,
    "ddpg": DDPG,
    "ddpg_plus_bc": DDPGPlusBC,
    "iql": IQL,
    "mopo": MOPO,
    "plas": PLASWithPerturbation,
    "sac": SAC,
    "td3": TD3,
    "td3_plus_bc": TD3PlusBC,
    "td3_plus_bc_aug": TD3PlusBCAug,
    "random": RandomPolicy,
}


def get_algo(name: str, discrete: bool) -> Type[AlgoBase]:
    """Returns algorithm class from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.

    Returns:
        type: algorithm class.

    """
    if discrete:
        if name in DISCRETE_ALGORITHMS:
            return DISCRETE_ALGORITHMS[name]
        raise ValueError(f"{name} does not support discrete action-space.")
    if name in CONTINUOUS_ALGORITHMS:
        return CONTINUOUS_ALGORITHMS[name]
    raise ValueError(f"{name} does not support continuous action-space.")


def create_algo(name: str, discrete: bool, **params: Any) -> AlgoBase:
    """Returns algorithm object from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.
        params (any): arguments for algorithm.

    Returns:
        d3rlpy.algos.base.AlgoBase: algorithm.

    """
    return get_algo(name, discrete)(**params)
