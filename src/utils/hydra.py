import functools
from dataclasses import dataclass
from typing import Any, Callable, List
from hydra.utils import get_method



@dataclass
class PartialWrapper:
    methods: List[Callable]

    def __call__(self, inputs) -> Any:
        for method in self.methods:
            inputs = method(inputs)
        return inputs


def partial(_partial_, *args, **kwargs):
    if isinstance(_partial_, list):
        methods = PartialWrapper([get_method(p) for p in _partial_])
        return methods
    return functools.partial(get_method(_partial_), *args, **kwargs)