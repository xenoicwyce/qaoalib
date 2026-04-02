import numpy as np
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate


def generate_sum_x_pauli_str(length):
    ret = []
    for i in range(length):
        paulis = ['I'] * length
        paulis[i] = 'X'
        ret.append(''.join(paulis))

    return ret

def qaoa(problem_ham: SparsePauliOp, reps: int = 1) -> QuantumCircuit:
    r"""
    Input:
    - problem_ham: Problem Hamiltonian to construct the QAOA circuit.
    Standard procedure would be:
    ```
        hamiltonian, offset = qubo.to_ising()
        qc = qaoa_circuit_from_qubo(hamiltonian)
    ```

    Returns:
    - qc: A QuantumCircuit object representing the QAOA circuit e^{-i\beta H_M} e^{-i\gamma H_C}.
    """
    num_qubits = problem_ham.num_qubits

    gamma = ParameterVector(name=r'$\gamma$', length=reps)
    beta = ParameterVector(name=r'$\beta$', length=reps)

    mixer_ham = SparsePauliOp(generate_sum_x_pauli_str(num_qubits))

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for p in range(reps):
        exp_gamma = PauliEvolutionGate(problem_ham, time=gamma[p])
        exp_beta = PauliEvolutionGate(mixer_ham, time=beta[p])
        qc.append(exp_gamma, qargs=range(num_qubits))
        qc.append(exp_beta, qargs=range(num_qubits))

    return qc

def multi_angle_qaoa(problem_ham: SparsePauliOp, reps: int = 1) -> QuantumCircuit:
    r"""
    Input:
    - problem_ham: Problem Hamiltonian to construct the QAOA circuit.
    Standard procedure would be:
    ```
        hamiltonian, offset = qubo.to_ising()
        qc = qaoa_circuit_from_qubo(hamiltonian)
    ```

    Returns:
    - qc: A QuantumCircuit object representing the QAOA circuit e^{-i\beta H_M} e^{-i\gamma H_C}.
    """
    num_qubits = problem_ham.num_qubits

    gamma = ParameterVector(name=r'$\gamma$', length=len(problem_ham) * reps)
    beta = ParameterVector(name=r'$\beta$', length=num_qubits * reps)

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for p in range(reps):
        for idx, term in enumerate(problem_ham):
            exp_gamma = PauliEvolutionGate(term, time=gamma[p * num_qubits + idx])
            qc.append(exp_gamma, qargs=range(num_qubits))
        for i in range(num_qubits):
            qc.rx(beta[p * num_qubits + i], i)

    return qc

def two_local(num_qubits: int, reps: int = 1, entanglement: str = 'linear') -> QuantumCircuit:
    # only linear or circular entanglement
    if entanglement not in ['linear', 'circular']:
        raise ValueError(f'Entanglment must be linear or circular, got {entanglement} instead.')

    qc = QuantumCircuit(num_qubits)
    params = ParameterVector('θ', num_qubits * (reps + 1))

    for i in range(num_qubits):
        qc.ry(params[i], i)

    for r in range(1, reps + 1):
        for i in range(num_qubits - 1):
            qc.cz(i, i+1)
        if entanglement == 'circular':
            qc.cz(num_qubits - 1, 0)
        for i in range(num_qubits):
            qc.ry(params[r * num_qubits + i], i)

    return qc

def _det_warm_start_angle(ci: float, epsilon: float):
    if ci <= epsilon:
        return 2 * np.arcsin(np.sqrt(epsilon))
    elif ci >= 1 - epsilon:
        return 2 * np.arcsin(np.sqrt(1 - epsilon))
    else:
        return 2 * np.arcsin(np.sqrt(ci))

def warm_start_qaoa(
    problem_ham: SparsePauliOp,
    relaxed_qp_solution: list[float],
    depth: int = 1,
    regularization: float = 0.01,
    flip_solution_indices: bool = False,
) -> QuantumCircuit:
    r"""
    Input:
    - problem_ham: Problem Hamiltonian to construct the QAOA circuit.
    - relaxed_qp_solution: The solution for the relaxed quadratic program.

    Returns:
    - qc: A QuantumCircuit object representing the QAOA circuit e^{-i\beta H_M} e^{-i\gamma H_C}.
    """
    num_qubits = problem_ham.num_qubits
    if num_qubits != len(relaxed_qp_solution):
        raise ValueError(f'The number of qubits and the length of the solution must be equal.')

    # try setting this to True if doesn't work, as Qiskit works with reversed indices.
    if flip_solution_indices:
        relaxed_qp_solution = relaxed_qp_solution[::-1]

    gamma = ParameterVector(name=r'$\gamma$', length=depth)
    beta = ParameterVector(name=r'$\beta$', length=depth)

    # calculate the thetas for warm start
    reg_warm_start_angle = lambda ci: _det_warm_start_angle(ci, regularization)
    theta = list(map(reg_warm_start_angle, relaxed_qp_solution))

    qc = QuantumCircuit(num_qubits)

    # initial layer
    for i in range(num_qubits):
        qc.ry(theta[i], i)

    for p in range(depth):
        # phase separation
        exp_gamma = PauliEvolutionGate(problem_ham, time=gamma[p])
        qc.append(exp_gamma, qargs=range(num_qubits))

        # mixer
        for i in range(num_qubits):
            qc.ry(-theta[i], i)
            qc.rz(beta[p], i)
            qc.ry(theta[i], i)

    return qc

def rzy_gate(theta: float) -> QuantumCircuit:
    """
    Add RZY gate to QuantumCircuit object, with Z acting on q1 and Y acting on q0 (following qiskit convention).
    RZY(theta) = exp(-i * theta * kron(Y, Z)).
    """
    qc = QuantumCircuit(2, name='Rzy')
    qc.rx(np.pi/2, 1)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.rx(-np.pi/2, 1)
    return qc

def drop_weights(G: nx.Graph) -> nx.Graph:
    G_unweighted = G.copy()
    for _, _, data in G_unweighted.edges(data=True):
        data['weights'] = 1.0

    return G_unweighted

def ihva_from_graph(G: nx.Graph, reps: int = 1, weighted_gates: bool = False) -> QuantumCircuit:
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    edge_buffer = []
    G_unweighted = drop_weights(G)
    while G_unweighted.edges:
        tree = nx.maximum_spanning_tree(G_unweighted)
        edge_buffer += list(tree.edges)
        G_unweighted.remove_edges_from(tree.edges)

    qc = QuantumCircuit(num_nodes)
    qc.h(range(num_nodes))
    params = ParameterVector('θ', num_edges * reps)
    for r in range(reps):
        for idx, edge in enumerate(edge_buffer):
            if r % 2 == 0:
                i, j = edge
            else:
                j, i = edge
            gate_weight = G[i][j]['weight'] if weighted_gates else 1.0
            qc.append(rzy_gate(gate_weight * params [r * num_edges + idx] ), qargs=[i, j])

    return qc

def linear_ansatz(num_qubits: int, reps: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector('θ', num_qubits * reps)

    # initial state
    qc.h(range(num_qubits - 1)) # leave the last qubit with state |0>

    for r in range(reps):
        for i in range(num_qubits):
            qc.ry(params[r * num_qubits + i], i)

    return qc

def pure_rx_ansatz(num_qubits: int) -> QuantumCircuit:
    params = ParameterVector('θ', num_qubits)
    qc = QuantumCircuit(num_qubits)

    for i in range(num_qubits):
        qc.rx(params[i], i)

    return qc