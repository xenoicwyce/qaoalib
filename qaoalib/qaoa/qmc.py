import numpy as np
from qiskit import Aer, execute

from .qis import qaoa_circuit, sv_backend
from .utils import expectation, I, Z
from ..math import fast_kron

class Qmc:
    def __init__(self, G, params):
        self.graph = G
        self.params = params
        self.circuit = qaoa_circuit(self.graph, self.params)
        self.result = None
        self.expectation = None

    def run(self, backend_name='qasm_simulator', **execute_kw):
        qc = self.circuit
        if backend_name == 'qasm_simulator':
            qc.measure_all()

        backend = Aer.get_backend(backend_name)
        job = execute(self.circuit, backend, **execute_kw)
        result = job.result()
        if backend_name == 'qasm_simulator':
            self.result = result.get_counts()
        elif backend_name == 'statevector_simulator':
            self.result = np.asarray(result.get_statevector())

        self.expectation = expectation(self.graph, self.result)

class QmcFastKron(Qmc):
    def __init__(self, G, params):
        super().__init__(G, params)

    def _fast_kron_exp(self, sv):
        sum_ = 0
        for u, v, d in self.graph.edges(data=True):
            edge = (u,v)
            kron_list = [Z if i in edge else I for i in range(len(self.graph.nodes))]
            kron_list.reverse()
            sum_ += d.get("weight",1) * (1 - (sv.conj().T @ fast_kron(kron_list, sv)).item().real)
            
 
        # for edge in self.graph.edges:
        #     kron_list = [Z if i in edge else I for i in range(len(self.graph.nodes))]
        #     kron_list.reverse()
        #     sum_ += (sv.conj().T @ fast_kron(kron_list, sv)).item().real

        # return a single expectation value
        return sum_/2

    def run(self, **execute_kw):
        job = execute(self.circuit, sv_backend, **execute_kw) # currently only support sv type
        self.result = job.result().get_statevector()
        self.expectation = self._fast_kron_exp(self.result)
