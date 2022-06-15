import numpy as np


# globals
I = np.eye(2)
X = np.array([
    [0, 1],
    [1, 0],
])
Z = np.array([
    [1, 0],
    [0, -1],
])
cnot = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
])
plus = np.ones((2, 1)) * 1/np.sqrt(2)

# functions
def split_gb(params):
    depth = len(params)//2
    gamma = params[:depth]
    beta = params[depth:]
    return gamma, beta

def random_qaoa_params(p, grange=None, brange=None):
    """Generate random QAOA parameters."""
    if grange is None:
        grange = (0, 2*np.pi)
    if brange is None:
        brange = (0, np.pi)

    rng = np.random.default_rng()
    # rng.uniform(low=0., high=1., size=None)
    grand = rng.uniform(grange[0], grange[1], p)
    brand = rng.uniform(brange[0], brange[1], p)
    return np.hstack([grand, brand])

def interp(old_params, new_params):
    gamma, beta = split_gb(old_params)
    new_gamma, new_beta = split_gb(new_params)
    return np.hstack((gamma, new_gamma, beta, new_beta)).tolist()

def interp_rand(params, grange=None, brange=None):
    """
    Adds a new random angle to the params array, and return
    the params required by (p+1).
    """
    if grange is None:
        grange = (0, 2*np.pi)
    if brange is None:
        brange = (0, np.pi)

    depth = len(params)//2
    gamma = params[:depth]
    beta = params[depth:]

    rng = np.random.default_rng()
    grand = rng.uniform(grange[0], grange[1], 1)
    brand = rng.uniform(brange[0], brange[1], 1)
    return np.hstack([gamma, grand, beta, brand])

def rx(theta):
    return np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)],
    ])

def rz(theta):
    return np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)],
    ])

def make_params_vec(gamma1d, beta1d, prev_params=None):
    if prev_params is None:
        return zip(gamma1d, beta1d)

    npts = gamma1d.shape[0]
    depth = len(prev_params)//2
    prev_gamma = prev_params[:depth]
    prev_beta = prev_params[depth:]

    gamma_list = (np.zeros((npts, depth)) + np.array(prev_gamma)).T.tolist()
    beta_list = (np.zeros((npts, depth)) + np.array(prev_beta)).T.tolist()

    return zip(*gamma_list, gamma1d, *beta_list, beta1d)

def ht_expectation(sv):
    zero, one = np.split(sv, 2)
    zero_prob = np.sum(np.abs(zero)**2)
    one_prob = np.sum(np.abs(one)**2)
    return zero_prob - one_prob

def _cut_value(G, eigenstate):
    eigenstate = eigenstate[::-1]
    cut = 0
    for u, v, d in G.edges(data=True):
        if eigenstate[u] != eigenstate[v]:
            w = d.get('weight', 1)
            cut += w
    return cut

def _sv2dict(sv):
    num_qubits = np.log2(sv.shape[0])
    if num_qubits % 1:
        raise ValueError('Input vector is not a valid statevector.')
    num_qubits = int(num_qubits)

    sv_dict = {
        f'{idx:0{num_qubits}b}': sv[idx]
        for idx in range(2**num_qubits)
    }
    return sv_dict

def expectation(G, counts_or_sv):
    sum_ = 0
    if isinstance(counts_or_sv, dict):
        counts = counts_or_sv
        total = sum(counts.values())
        for eigs, count in counts.items():
            sum_ += _cut_value(G, eigs) * count / total
        return sum_

    elif isinstance(counts_or_sv, np.ndarray):
        sv = _sv2dict(counts_or_sv)
        for eigs, count in sv.items():
            sum_ += _cut_value(G, eigs) * (np.abs(sv[eigs])**2)
        return sum_

    else:
        raise TypeError(f'Unrecognized type {type(counts_or_sv)}.')
