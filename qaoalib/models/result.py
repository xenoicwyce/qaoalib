import numpy as np
import networkx as nx
import json

from pydantic import BaseModel, Field
from pathlib import Path
from collections import defaultdict

from qaoalib.utils import maxcut_brute, graph_from_name
from qaoalib.json import to_serializable

ddl_wrapper = lambda : defaultdict(list)


class BaseResult(BaseModel):
    name: str
    true_obj: int = 0

    def solve_brute(self) -> None:
        G = graph_from_name(self.name)
        self.true_obj, _ = maxcut_brute(G)

    def solve_gurobi(self) -> None:
        from qiskit_optimization.algorithms import GurobiOptimizer
        from qiskit_optimization.applications import Maxcut
        
        G = graph_from_name(self.name)
        w = nx.adjacency_matrix(G)
        maxcut = Maxcut(w)
        qp = maxcut.to_quadratic_program()

        grb_result = GurobiOptimizer().solve(qp)
        self.true_obj = int(grb_result.fval)

    def dump(self, target_dir='.') -> None:
        target_dir = Path(target_dir)
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
        with open(target_dir/f'{self.name}.json', 'w') as f:
            json.dump(self.dict(), f, default=to_serializable)


class SingleTrialResult(BaseResult):
    expectations: dict[int, float] = Field(default_factory=dict)
    nfevs: dict[int, int] = Field(default_factory=dict)
    initial_params: dict[int, list[float]] = Field(default_factory=dict)
    opt_params: dict[int, list[float]] = Field(default_factory=dict)
    alphas: dict[int, float] = Field(default_factory=dict)

    def compute_alpha(self) -> None:
        for p, exp in self.expectations.items():
            self.alphas[p] = exp / self.true_obj


class TQAResult(SingleTrialResult):
    delta_t: dict[int, float] = Field(default_factory=dict)


class MultipleTrialResult(BaseResult):
    expectations: dict[int, list[float]] = Field(default_factory=ddl_wrapper)
    nfevs: dict[int, list[int]] = Field(default_factory=ddl_wrapper)
    initial_params: dict[int, list[list[float]]] = Field(default_factory=ddl_wrapper)
    opt_params: dict[int, list[list[float]]] = Field(default_factory=ddl_wrapper)
    alphas: dict[int, list[float]] = Field(default_factory=ddl_wrapper)

    def compute_alpha(self) -> None:
        for p, exps in self.expectations.items():
            self.alphas[p] = (np.asarray(exps) / self.true_obj).tolist()

    def init_defaultdict(self) -> None:
        """ Initialize defaultdict for the attributes when loaded from file. """
        for attr in ['expectations', 'nfevs', 'initial_params', 'opt_params', 'alphas']:
            dd = defaultdict(list, getattr(self, attr))
            setattr(self, attr, dd)


class ItlwResult(MultipleTrialResult):
    pass

class VQEResult(BaseResult):
    expectations: list[float] = Field(default_factory=list)
    nfevs: list[int] = Field(default_factory=list)
    initial_params: list[list[float]] = Field(default_factory=list)
    opt_params: list[list[float]] = Field(default_factory=list)
    alphas: list[float] = Field(default_factory=list)