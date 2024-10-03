from qiskit import QuantumRegister, QuantumCircuit
from qiskit import Aer, execute


sv_backend = Aer.get_backend('statevector_simulator')
aer_backend = Aer.get_backend('aer_simulator')

def qaoa_circuit(G, params):
    """
    QAOA circuit for Max-cut \sum(I-ZZ). No longer follow qiskit QAOA circuit.
    If want to reproduce the circuit used in qiskit \sum(-ZZ), substitute
    gamma as -gamma will do.
    """
    depth = len(params)//2
    gamma = params[:depth]
    beta = params[depth:]

    q = QuantumRegister(len(G.nodes))
    qc = QuantumCircuit(q)

    qc.h(q)
    for p in range(depth):
        for u, v, d in G.edges(data=True):
            weight = d.get('weight', 1)
            qc.cx(u, v)
            qc.rz(-gamma[p]*weight, v)
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

        # bit reversal is required to calculate probabilities
        qc = qc.reverse_bits()
        qc_list.append(qc)

    return qc_list

def run_many_circuits(qc_list):
    job = execute(qc_list, sv_backend)
    result = job.result()
    # return a list of np.ndarray of statevectors
    return [result.get_statevector(idx) for idx in range(len(qc_list))]
