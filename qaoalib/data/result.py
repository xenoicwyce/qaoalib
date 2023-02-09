from typing import Union, Any
from typing_extensions import TypeAlias
from pydantic import BaseModel

from graph import RegGraphData, GnpGraphData

SingleTrialDict: TypeAlias = dict[int, Any]
MultipleTrialDict: TypeAlias = dict[int, list[Any]]


class Result(BaseModel):
    graph: Union[RegGraphData, GnpGraphData]
    expectations: SingleTrialDict
    alpha: SingleTrialDict
    initial_params: SingleTrialDict
    opt_params: SingleTrialDict
    nfev: SingleTrialDict


class PfResult(BaseModel):
    graph: Union[RegGraphData, GnpGraphData]
    expectations: MultipleTrialDict
    alpha: MultipleTrialDict
    initial_params: MultipleTrialDict
    opt_params: MultipleTrialDict
    nfev: MultipleTrialDict
