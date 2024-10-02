import numpy as np
from typing import TYPE_CHECKING, Optional, Union, Sequence
from typing_extensions import TypeAlias
# from .params import Params

if TYPE_CHECKING:
    import numpy.typing as npt

Params: TypeAlias = Union[Sequence[float], "npt.NDArray[np.float_]"]


def interp():
    ...

def bilinear(
    prev_params: Params,
    pp_params: Params,
    gamma_bound: Optional[tuple[float, float]] = None,
    beta_bound: Optional[tuple[float, float]] = None,
) -> "npt.NDArray[np.float_]":
    if gamma_bound is None:
        gamma_bound = (0, np.pi)
    if beta_bound is None:
        beta_bound = (0, np.pi/2)

    prev_params = np.asarray(prev_params)
    pp_params = np.asarray(pp_params) # prev_prev_params (p-2)

    # split into gamma and beta
    prev_gb = np.split(prev_params, 2)
    pp_gb = np.split(pp_params, 2)

    new_gb = []
    for i in range(2):
        delta2 = prev_gb[i][-2] - pp_gb[i][-1]
        pp_gb[i] = np.hstack([pp_gb[i], prev_gb[i][-1] - delta2])

        diff = prev_gb[i] - pp_gb[i]
        new_x = prev_gb[i] + diff

        delta3 = new_x[-1] - new_x[-2]
        new_x = np.hstack([new_x, new_x[-1] + delta3])

        if i == 0:
            new_x = np.clip(new_x, *gamma_bound)
        else:
            new_x = np.clip(new_x, *beta_bound)

        new_gb.append(new_x)

    return np.hstack(new_gb)


def params_fixing(
    prev_params: Params,
    gamma_bound: Optional[tuple[float, float]] = None,
    beta_bound: Optional[tuple[float, float]] = None,
) -> "npt.NDArray[np.float_]":
    if gamma_bound is None:
        gamma_bound = (0, np.pi)
    if beta_bound is None:
        beta_bound = (0, np.pi/2)

    prev_params = np.asarray(prev_params)
    gamma, beta = np.split(prev_params)

    gamma = np.hstack([gamma, np.random.uniform(*gamma_bound)])
    beta = np.hstack([beta, np.random.uniform(*beta_bound)])

    return np.hstack([gamma, beta])
