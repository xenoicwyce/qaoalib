from typing import Sequence
from pydantic import BaseModel


class GraphData(BaseModel):
    n_nodes: int
    edges: list[Sequence[int]]
    n_edges: int
    shift: float
    true_obj: int


class RegGraphData(GraphData):
    deg: int


class GnpGraphData(GraphData):
    prob: float
