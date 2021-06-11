import numpy as np

from .base import QmcLandscapeBase
from ..utils import interp, ht_expectation
from ..qis import hadamard_test_circuits, run_many_circuits


class HadamardTest(QmcLandscapeBase):
    def __init__(self, G, prev_params=None):
        super().__init__(G, prev_params)

    def get_circuits(self, params):
        return hadamard_test_circuits(self.graph, params)

    def expectation_grid(self, grange, brange, npts, prev_params=None):
        qc_list = []
        for beta in brange:
            for gamma in grange:
                if prev_params is None:
                    qc_list += self.get_circuits([gamma, beta])
                else:
                    qc_list += self.get_circuits(interp(prev_params, [gamma, beta]))

        sv_list = run_many_circuits(qc_list)
        exp_arr = np.array(list(map(ht_expectation, sv_list)))\
                    .reshape((npts*npts, len(self.edge_list)))
        exp_arr = np.sum(exp_arr, axis=1).reshape((npts, npts))
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
