import numpy as np

from .base import QmcLandscapeBase
from ..utils import I, Z, plus
from ..utils import rx, make_params_vec


class DirectNumpy(QmcLandscapeBase):
    def __init__(self, G, prev_params=None):
        super().__init__(G, prev_params)
        self.plusxn = self._mplus()
        self.hamiltonian = self._hmt()

    def _mplus(self):
        ans = plus
        for q in range(1, self.num_qubits):
            ans = np.kron(ans, plus)
        return ans

    def _hmt(self):
        N = 2 ** self.num_qubits
        ans = np.zeros((N, N))
        for u, v in self.edge_list:
            ans += np.eye(N) - self.tensor(Z, [u, v])
        return ans/2

    def tensor(self, u3, qubits):
        if 0 in qubits:
            ans = u3
        else:
            ans = I

        for idx in range(1, self.num_qubits):
            if idx in qubits:
                ans = np.kron(ans, u3)
            else:
                ans = np.kron(ans, I)
        return ans

    def ucost(self, q1, q2, gamma):
        return np.diag(np.exp(1j*gamma/2*np.diag(np.eye(2**self.num_qubits)-self.tensor(Z, [q1, q2]))))

    def umixer_all(self, beta):
        return self.tensor(rx(2*beta), list(range(self.num_qubits)))

    def ansatz(self, gamma_vec, beta_vec):
        ans = self.plusxn
        for gamma, beta in zip(gamma_vec, beta_vec):
            for u, v in self.edge_list:
                ans = self.ucost(u, v, gamma) @ ans
            ans = self.umixer_all(beta) @ ans
        return ans

    def expectation(self, params):
        if self.plusxn is None or self.hamiltonian is None:
            self.plusxn = self._mplus()
            self.hamiltonian = self._hmt()
        depth = len(params)//2
        gamma_vec = params[:depth]
        beta_vec = params[depth:]
        ansatz = self.ansatz(gamma_vec, beta_vec)
        ans = ansatz.conj().T @ self.hamiltonian @ ansatz
        ans = ans[0][0]
        if np.isclose(ans.real, np.abs(ans)):
            return ans.real
        else:
            return ans

    def create_grid(self, npts=100, gmin=0, gmax=2*np.pi, bmin=0, bmax=np.pi):
        grange = np.linspace(gmin, gmax, npts)
        brange = np.linspace(bmin, bmax, npts)
        gmesh, bmesh = np.meshgrid(grange, brange)
        gg = gmesh.reshape((-1,))
        bb = bmesh.reshape((-1,))

        exp_arr = np.array(list(map(self.expectation, make_params_vec(gg, bb, self.prev_params))))\
                        .reshape((npts, npts))

        self.npts = npts
        self.gmesh = gmesh
        self.bmesh = bmesh
        self.exp_arr = exp_arr
