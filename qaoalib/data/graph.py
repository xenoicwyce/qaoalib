import networkx as nx
from typing import Sequence, Optional, Union, Literal
from pydantic import BaseModel

from ..utils import maxcut_brute

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


def convert_graph(
    G: nx.Graph,
    graph_type: Literal['reg', 'gnp'],
    prob: Optional[float] = None,
    solve_brute: bool = True,
) -> Union[RegGraphData, GnpGraphData]:

    if solve_brute:
        true_obj, _ = maxcut_brute(G)
    else:
        true_obj = 0

    n_nodes = len(G.nodes)
    edges = list(G.edges)
    n_edges = len(G.edges)
    shift = -n_edges / 2.0

    if graph_type == 'reg':
        deg = len(list(G.neighbors(0)))
        return RegGraphData(
            deg=deg,
            n_nodes=n_nodes,
            edges=edges,
            n_edges=n_edges,
            shift=shift,
            true_obj=true_obj,
        )

    elif graph_type == 'gnp':
        if prob is None:
            raise ValueError('`prob` must be passed for Gnp graphs.')
        return GnpGraphData(
            prob=prob,
            n_nodes=n_nodes,
            edges=edges,
            n_edges=n_edges,
            shift=shift,
            true_obj=true_obj,
        )
