import numpy as np
import json

from pydantic import BaseModel, Field
from pathlib import Path
from collections import defaultdict

from qaoalib.utils import maxcut_brute, graph_from_name

ddl_wrapper = lambda : defaultdict(list)


class BaseResult(BaseModel):
    name: str
    true_obj: int = 0

    def solve_brute(self) -> None:
        G = graph_from_name(self.name)
        self.true_obj, _ = maxcut_brute(G)

    def dump(self, target_dir='.') -> None:
        target_dir = Path(target_dir)
        with open(target_dir/f'{self.name}.json', 'w') as f:
            json.dump(self.dict(), f)

    def init_defaultdict(self) -> None:
        """ Initialize defaultdict for the attributes when loaded from file. """
        for attr in ['expectations', 'nfevs', 'initial_params', 'opt_params', 'alphas']:
            dd = defaultdict(list, getattr(self, attr))
            setattr(self, attr, dd)


class SingleTrialResult(BaseResult):
    expectations: dict[int, float] = Field(default_factory=ddl_wrapper)
    nfevs: dict[int, int] = Field(default_factory=ddl_wrapper)
    initial_params: dict[int, list[float]] = Field(default_factory=ddl_wrapper)
    opt_params: dict[int, list[float]] = Field(default_factory=ddl_wrapper)
    alphas: dict[int, float] = Field(default_factory=ddl_wrapper)

    def compute_alpha(self) -> None:
        for p, exp in self.expectations.items():
            self.alphas[p] = exp / self.true_obj


class MultipleTrialResult(BaseResult):
    expectations: dict[int, list[float]] = Field(default_factory=ddl_wrapper)
    nfevs: dict[int, list[int]] = Field(default_factory=ddl_wrapper)
    initial_params: dict[int, list[list[float]]] = Field(default_factory=ddl_wrapper)
    opt_params: dict[int, list[list[float]]] = Field(default_factory=ddl_wrapper)
    alphas: dict[int, list[float]] = Field(default_factory=ddl_wrapper)

    def compute_alpha(self) -> None:
        for p, exps in self.expectations.items():
            self.alphas[p] = (np.asarray(exps) / self.true_obj).tolist()


class ItlwResult(BaseResult):
    expectations: dict[int, list[float]] = Field(default_factory=ddl_wrapper)
    nfevs: dict[int, list[int]] = Field(default_factory=ddl_wrapper)
    initial_params: dict[int, list[list[float]]] = Field(default_factory=ddl_wrapper)
    opt_params: dict[int, list[list[float]]] = Field(default_factory=ddl_wrapper)
    alphas: dict[int, list[float]] = Field(default_factory=ddl_wrapper)

    def compute_alpha(self) -> None:
        for p, exps in self.expectations.items():
            self.alphas[p] = (np.asarray(exps) / self.true_obj).tolist()
