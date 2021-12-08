from qiskit import Aer, execute

from .qis import qaoa_circuit
from .utils import expectation

class Qmc:
    def __init__(self, G, params):
        self.graph = G
        self.params = params
        self.circuit = qaoa_circuit(self.graph, self.params)
        self.result = None
        self.expectation = None

    def run(self, backend_name, **execute_kw):
        qc = self.circuit
        if backend_name == 'qasm_simulator':
            qc.measure_all()

        backend = Aer.get_backend(backend_name)
        job = execute(self.circuit, backend, **execute_kw)
        result = job.result()
        if backend_name == 'qasm_simulator':
            self.result = result.get_counts()
        elif backend_name == 'statevector_simulator':
            self.result = result.get_statevector()

        self.expectation = expectation(self.graph, self.result)
