import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer, execute


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

sv_backend = Aer.get_backend('statevector_simulator')


# functions
def split_gb(params):
    depth = len(params)//2
    gamma = params[:depth]
    beta = params[depth:]
    return gamma, beta

def random_qaoa_params(p):
    """Generate random QAOA parameters."""
    gamma = np.random.rand(p,) * 2*np.pi
    beta = np.random.rand(p,) * np.pi
    return np.hstack((gamma, beta))

def interp(old_params, new_params):
    gamma, beta = split_gb(old_params)
    new_gamma, new_beta = split_gb(new_params)
    return np.hstack((gamma, new_gamma, beta, new_beta))

def interp_rand(params):
    """
    Adds a new random angle to the params array, and return
    the params required by (p+1).
    """
    depth = len(params)//2
    gamma = params[:depth]
    beta = params[depth:]
    gamma = np.hstack((gamma, np.random.rand()*2*np.pi))
    beta = np.hstack((beta, np.random.rand()*np.pi))
    return np.hstack((gamma, beta))

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

def qaoa_circuit(G, params):
    depth = len(params)//2
    gamma = params[:depth]
    beta = params[depth:]

    q = QuantumRegister(len(G.nodes))
    qc = QuantumCircuit(q)

    qc.h(q)
    for p in range(depth):
        for u, v in G.edges:
            qc.cx(u, v)
            qc.rz(gamma[p], v)
            qc.cx(u, v)
        qc.rx(2*beta[p], q)
    return qc

def hadamard_test_circuits(G, params):
    """
    Return a list of QuantumCircuit's for hadamard tests.
    """
    qc_list = []
    ansatz_circ = qaoa_circuit(G, params)

    for u, v in G.edges:
        q = QuantumRegister(1)
        ansatz = QuantumRegister(ansatz_circ.num_qubits)
        qc = QuantumCircuit(q, ansatz)

        qc.h(q)
        qc.append(ansatz_circ, ansatz)
        qc.cz(q, ansatz[u])
        qc.cz(q, ansatz[v])
        qc.h(q)

        qc = qc.reverse_bits()
        qc_list.append(qc)

    return qc_list

def ht_expectation(sv):
    zero, one = np.split(sv, 2)
    zero_prob = np.sum(np.abs(zero)**2)
    one_prob = np.sum(np.abs(one)**2)
    return zero_prob - one_prob

def run_many_circuits(qc_list):
    job = execute(qc_list, sv_backend)
    result = job.result()
    # return a list of np.ndarray of statevectors
    return [result.get_statevector(idx) for idx in range(len(qc_list))]