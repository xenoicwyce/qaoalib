import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib import cm

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

def random_qaoa_params(p):
    """Generate random QAOA parameters."""
    gamma = np.random.rand(p,) * 2*np.pi
    beta = np.random.rand(p,) * np.pi
    return np.hstack((gamma, beta))

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

class QaoaMaxCut:
    def __init__(self, G, prev_params=None):
        self.graph = G
        self.num_qubits = len(G.nodes)
        self.edge_list = list(G.edges)
        self.plusxn = self._mplus()
        self.hamiltonian = self._hmt()
        self.prev_params = prev_params
        self.gmesh = None
        self.bmesh = None
        self.exp_arr = None
        self.depth = 1 if prev_params is None else len(prev_params)//2+1

    @classmethod
    def set_timer(cls):
        for attr in dir(cls):
            if not attr.startswith('__'):
                if callable(getattr(cls, attr)):
                    setattr(cls, attr, func_timer(getattr(cls, attr)))

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

    def cnot(self, ctrl, target):
        if ctrl >= self.num_qubits:
            raise ValueError('Control index out of range.')
        if target >= self.num_qubits:
            raise ValueError('Target index out of range.')

        ans = np.eye(2**self.num_qubits)
        for n in range(2**self.num_qubits):
            if to_bin(n, self.num_qubits)[ctrl] == '1':
                ans[n, n] = 0
                ans[n, n^((2**self.num_qubits) >> (target+1))] = 1
        return ans

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

        self.gmesh = gmesh
        self.bmesh = bmesh
        self.exp_arr = exp_arr

    def get_max(self):
        if self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        exp_max = np.max(self.exp_arr)
        whr = np.where(np.isclose(self.exp_arr, exp_max))
        indices = zip(whr[0], whr[1])
        angle_list = [(self.gmesh[idx], self.bmesh[idx]) for idx in indices]
        return (exp_max, angle_list)

    def show_landscape(self, plot_options={}):
        defaults = {
            'figsize': (16, 9),
        }
        defaults.update(plot_options)
        figsize = defaults['figsize']

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=figsize)

        if self.gmesh is None or self.bmesh is None or self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        surf = ax.plot_surface(self.gmesh, self.bmesh, self.exp_arr, cmap=cm.coolwarm)
        ax.set_xlabel('gamma')
        ax.set_ylabel('beta')
        ax.set_zlabel('expectation')
        fig.colorbar(surf, shrink=.5)

        plt.show()

    def show_heatmap(self, plot_options={}):
        if self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        plt.xlabel('gamma_p/2pi')
        plt.ylabel('beta_p/pi')
        plt.imshow(self.exp_arr, cmap=cm.coolwarm, origin='lower', extent=[0, 1, 0, 1])

