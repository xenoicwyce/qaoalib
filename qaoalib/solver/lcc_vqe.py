import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_optimization import QuadraticProgram

from .vqe import VQE

class LCCVQE(VQE):
    """
    LCC for VQE ansatz with circular entanglement.
    * Only works for one-local (Z) or two-local (ZZ) operations.
    * Currently only consider the TwoLocal ansatz with RY rotation, CZ entanglement, and reps=1.
    """

    def __init__(
        self,
        quadratic_program: QuadraticProgram,
        shots: int = None,
        sampler=None,
        estimator=None,
        optimizer=None,
    ) -> None:
        super().__init__(
            quadratic_program,
            reps=1,
            shots=shots,
            sampler=sampler,
            estimator=estimator,
            optimizer=optimizer,
        )

    @staticmethod
    def get_pauli_indices(pauli: Pauli) -> list[int]:
        return np.argwhere(pauli.z).reshape(-1).tolist()

    @staticmethod
    def generate_local_ansatz(num_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        theta = ParameterVector('θ', 2 * num_qubits)

        # First layer RY
        for k in range(num_qubits):
            qc.ry(theta[k], k)

        # Entangling CNOT
        if num_qubits == 6:
            for j, k in [(0, 1), (1, 2), (3, 4), (4, 5)]:
                qc.cz(j, k)
        else:
            for k in range(num_qubits - 1):
                qc.cz(k, k + 1)

        # Second layer RY
        for k in range(num_qubits):
            qc.ry(theta[num_qubits + k], k)

        return qc

    def _distance(self, i, j):
        return min(abs(i - j), abs(i + self.num_qubits - j), abs(j - i + self.num_qubits))

    def generate_pubs(
        self,
        params: list[float] | np.ndarray,
    ) -> list[tuple[QuantumCircuit, SparsePauliOp, np.ndarray]]:
        pubs = []

        for observable in self.hamiltonian:
            pauli_indices = self.get_pauli_indices(observable.paulis[0])
            if len(pauli_indices) == 1:
                # one-local, 3-qubit
                qc = self.generate_local_ansatz(3)
                local_ob = SparsePauliOp('IZI', observable.coeffs[0])
                i = pauli_indices[0]
                first_layer = np.array([(i - 1), i, (i + 1)]) % self.num_qubits

            elif len(pauli_indices) == 2:
                # two-local
                i, j = pauli_indices
                if self._distance(i, j) == 1:
                    # 4-qubit

                    qc = self.generate_local_ansatz(4)
                    local_ob = SparsePauliOp('IZZI', observable.coeffs[0])

                    if i == 0 and j == self.num_qubits - 1:
                        first_layer = np.array([j - 1, j, i, i + 1]) % self.num_qubits
                    else:
                        first_layer = np.array([i - 1, i, j, j + 1]) % self.num_qubits

                elif self._distance(i, j) == 2:
                    # 5-qubit
                    qc = self.generate_local_ansatz(5)
                    local_ob = SparsePauliOp('IZIZI', observable.coeffs[0])

                    if i + 2 == j:
                        first_layer = np.array([i - 1, i, i + 1, j, j + 1]) % self.num_qubits
                    else:
                        first_layer = np.array([j - 1, j, i - 1, i, i + 1]) % self.num_qubits

                elif self._distance(i, j) > 2:
                    # 6-qubit
                    qc = self.generate_local_ansatz(6)
                    local_ob = SparsePauliOp('IZIIZI', observable.coeffs[0])
                    first_layer = np.array([i - 1, i, i + 1, j - 1, j, j + 1]) % self.num_qubits
            else:
                raise ValueError(f'Pauli indices must be of 1 or 2 length. Got {len(pauli_indices)} instead.')

            second_layer = first_layer + self.num_qubits
            param_indices = np.hstack([first_layer, second_layer]).tolist()

            # construct pub
            params = np.asarray(params)
            pubs.append((qc, local_ob, params[param_indices]))

        return pubs