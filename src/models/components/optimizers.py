from typing import Any, Callable, Dict, Optional

import torch

from src.utils import helpers


class BaseOptimizer(torch.optim.Optimizer):
    def __int__(self, args):
        self.optimizer = helpers.get_optimizer(args)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = ...) -> None:
        self.optimizer.zero_grad()
