"""
Taken and modified from:
https://gist.github.com/ahwillia/f65bc70cb30206d4eadec857b98c4065
"""

import numpy as np

def _unfold(tens, mode, dims):
    """
    Unfolds tensor into matrix.
    Parameters
    ----------
    tens : ndarray, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape
    Returns
    -------
    matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
    """
    if mode == 0:
        return tens.reshape(dims[0], -1)
    else:
        return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)


def _refold(vec, mode, dims):
    """
    Refolds vector into tensor.
    Parameters
    ----------
    vec : ndarray, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape
    Returns
    -------
    tens : ndarray, tensor with shape == dims
    """
    if mode == 0:
        return vec.reshape(dims)
    else:
        # Reshape and then move dims[mode] back to its
        # appropriate spot (undoing the `unfold` operation).
        tens = vec.reshape(
            [dims[mode]] +
            [d for m, d in enumerate(dims) if m != mode]
        )
        return np.moveaxis(tens, 0, mode)

def fast_kron(As, v):
    """
    Computes matrix-vector multiplication between
    matrix kron(As[0], As[1], ..., As[N]) and vector
    v without forming the full kronecker product.
    """
    dims = [A.shape[0] for A in As]
    vt = v.reshape(dims)
    for i, A in enumerate(As):
        vt = _refold(A @ _unfold(vt, i, dims), i, dims)
    return vt.ravel()
