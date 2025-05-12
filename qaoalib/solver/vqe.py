import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import (
    BackendSamplerV2 as BackendSampler,
    BackendEstimatorV2 as BackendEstimator,
)
from qiskit.quantum_info import SparsePauliOp, Pauli

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import GurobiOptimizer
from qiskit_algorithms.optimizers import COBYLA, OptimizerResult
from qiskit_aer import AerSimulator

TWO_PI = 2 * np.pi


class VQE:
    def __init__(
        self,
        quadratic_program: QuadraticProgram,
        ansatz: QuantumCircuit,
        shots: int = None,
        sampler=None,
        estimator=None,
        optimizer=None,
    ) -> None:
        self.qp = quadratic_program
        self.converter = QuadraticProgramToQubo()
        self.qubo = self.converter.convert(self.qp)

        hamiltonian, offset = self.qubo.to_ising()
        self.hamiltonian: SparsePauliOp = hamiltonian
        self.offset: float = offset

        self.num_qubits = hamiltonian.num_qubits
        self.ansatz = ansatz

        backend_sv = AerSimulator(method='statevector')
        backend_mps = AerSimulator(method='matrix_product_state')
        self.sampler = BackendSampler(backend=backend_mps) if sampler is None else sampler
        self.estimator = BackendEstimator(backend=backend_sv) if estimator is None else estimator
        self.optimizer = COBYLA() if optimizer is None else optimizer
        self.shots = shots

        self.optimal_params = None
        self.optimal_solution = None

    def generate_random_params(self, scale=TWO_PI):
        return np.random.rand(self.ansatz.num_parameters) * scale

    def generate_pubs(
        self,
        params: list[float] | np.ndarray,
    ) -> list[tuple[QuantumCircuit, SparsePauliOp, np.ndarray]]:
        return [(self.ansatz, self.hamiltonian, params)]

    def compute_energy(self, params: list[float] | np.ndarray) -> float:
        """
        Computes the sum of expectation of the ZZ terms.
        """
        pubs = self.generate_pubs(params)
        results = self.estimator.run(pubs).result()
        evs = [result.data.evs for result in results]
        return sum(evs)

    def _sample_optimal_circuit(self) -> dict[str, int]:
        """
        Run the full circuit with sampler to get the solution.
        """
        if self.optimal_params is None:
            raise ValueError('Problem not yet solved. Run LCCVQE.solve() to solve the problem.')

        ansatz = self.ansatz.copy()
        ansatz.measure_all()
        result = self.sampler.run([(ansatz, self.optimal_params)], shots=self.shots).result()[0]
        return result.data.meas.get_counts()

    def _sample_most_likely(self) -> list[int]:
        counts = self._sample_optimal_circuit()
        highest_count = max(counts.values())

        for bit_string, count in counts.items():
            if count == highest_count:
                return list(map(int, bit_string[::-1])) # flip the bit-string due to qiskit ordering

    def get_qp_solution(self) -> list[float]:
        qubo_solution = self._sample_most_likely()
        return self.converter.interpret(qubo_solution)

    def solve(
        self,
        initial_point: list[float] | np.ndarray = None,
        run_sampler: bool = False,
    ) -> OptimizerResult:
        """
        Calls the Scipy minimize function and returns the OptimizerResult object.
        """
        if initial_point is None:
            initial_point = self.generate_random_params()
        else:
            assert np.asarray(initial_point).shape[0] == self.ansatz.num_parameters, 'Parameter length does not match.'

        def obj_func(params):
            return self.compute_energy(params)

        result = self.optimizer.minimize(obj_func, initial_point)
        self.optimal_params = result.x

        if run_sampler:
            self.optimal_solution = self.get_qp_solution()

        return result

    def solve_gurobi(self) -> float:
        result = GurobiOptimizer().solve(self.qp)
        return result.fval