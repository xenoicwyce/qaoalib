import numpy as np

from .base import QmcLandscapeBase
from ..utils import I, Z, interp
from ..qis import qaoa_circuit, run_many_circuits
from ...math import fast_kron


class HybridFast(QmcLandscapeBase):
    def __init__(self, G, prev_params=None):
        super().__init__(G, prev_params)

    def get_circuit(self, params):
        return qaoa_circuit(self.graph, params)

    def _fast_kron_exp(self, sv):
        sum_ = 0
        for edge in self.edge_list:
            kron_list = [Z if i in edge else I for i in range(self.num_qubits)]
            kron_list.reverse()
            sum_ += (sv.conj().T @ fast_kron(kron_list, sv)).item().real
        # return a single expectation value
        return (len(self.edge_list) - sum_)/2

    def expectation_grid(self, grange, brange, npts, prev_params=None):
        if prev_params is None:
            qc_list = [self.get_circuit([gamma, beta]) for beta in brange for gamma in grange]
        else:
            qc_list = [self.get_circuit(interp(prev_params, [gamma, beta]))
                        for beta in brange for gamma in grange]
        ansatz_list = run_many_circuits(qc_list)
        exp_arr = np.array(list(map(self._fast_kron_exp, ansatz_list))).reshape((npts, npts))
        return exp_arr

    def create_grid(self, npts=100, gmin=0, gmax=2*np.pi, bmin=0, bmax=np.pi):
        grange = np.linspace(gmin, gmax, npts)
        brange = np.linspace(bmin, bmax, npts)
        gmesh, bmesh = np.meshgrid(grange, brange)

        exp_arr = self.expectation_grid(grange, brange, npts, self.prev_params)

        self.npts = npts
        self.gmesh = gmesh
        self.bmesh = bmesh
        self.exp_arr = exp_arr
