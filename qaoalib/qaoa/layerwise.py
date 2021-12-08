import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

from .utils import I, Z
from .utils import rx

class Layerwise:
    def __init__(self, G):
        self.graph = G
        self.num_qubits = len(G.nodes)
        self.edge_list = list(G.edges)
        self.hamiltonian = self.get_maxcut_hmt()
        self.maxcut = np.max(self.hamiltonian)
        self.best_ansatz = self.plusxn()
        self.best_params = None
        self.exp_arr = None
        self.npts = None
        self.gmesh = None
        self.bmesh = None
        self.max_exps = None
        self.depth = 0

    def tensor_prod(self, u3, qubits):
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

    def plusxn(self):
        N = 2 ** self.num_qubits
        return np.ones((N, 1))/np.sqrt(N)

    def get_maxcut_hmt(self):
        N = 2 ** self.num_qubits
        ans = np.zeros((N, N))
        for u, v in self.edge_list:
            ans += np.eye(N) - self.tensor_prod(Z, [u, v])
        return ans/2

    def ehz(self, gamma):
        eigs = np.diag(self.hamiltonian)
        return np.diag(np.exp(1j*gamma/2*eigs))

    def ehx(self, beta):
        return self.tensor_prod(rx(2*beta), list(range(self.num_qubits)))

    def ansatz(self, gamma, beta):
        ans = self.best_ansatz[:, -1][:, np.newaxis]
        return self.ehx(beta) @ self.ehz(gamma) @ ans

    def expectation(self, gamma, beta):
        right = self.ansatz(gamma, beta)
        left = right.conj().T
        return (left @ self.hamiltonian @ right).real

    def create_grid(self, npts, gmin=0, gmax=2*np.pi, bmin=0, bmax=np.pi):
        grange = np.linspace(gmin, gmax, npts)
        brange = np.linspace(bmin, bmax, npts)
        gmesh, bmesh = np.meshgrid(grange, brange)
        gg = gmesh.reshape((-1,))
        bb = bmesh.reshape((-1,))

        exp_arr = np.array(list(map(self.expectation, gg, bb)))\
                        .reshape((npts, npts))

        self.npts = npts
        self.gmesh = gmesh
        self.bmesh = bmesh
        if self.exp_arr is None:
            self.exp_arr = exp_arr[:, :, np.newaxis]
        else:
            self.exp_arr = np.dstack((self.exp_arr, exp_arr))

    # def get_max(self, p):
    #     if self.exp_arr is None:
    #         raise ValueError('Grid not found. Run create_grid() method first.')

    #     exp_arr = self.exp_arr[:, :, p-1]
    #     max_exp = np.max(exp_arr)
    #     whr = np.where(np.isclose(exp_arr, max_exp))
    #     indices = zip(whr[0], whr[1])
    #     angle_list = [(self.gmesh[idx], self.bmesh[idx]) for idx in indices]
    #     return (max_exp, angle_list)

    def find_args(self, p, value):
        """ Find the nearest args given a value. """

        if self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        dist_arr = np.abs(self.exp_arr[:, :, p-1] - value)
        nearest = np.min(dist_arr)
        whr = np.where(np.isclose(dist_arr, nearest))
        indices = zip(whr[0], whr[1])
        angle_list = [(self.gmesh[idx], self.bmesh[idx]) for idx in indices]
        return angle_list

    def run(self, p_end, npts=50, cutoff=1.0):
        for i in range(1, p_end + 1):
            print(f'Creating grid for p={i}')
            self.create_grid(npts)
            max_exp = np.max(self.exp_arr[:, :, i-1])
            best_params = self.find_args(i, cutoff * max_exp)[0] # take only one pair of angles

            if self.max_exps is None:
                self.max_exps = max_exp
            else:
                self.max_exps = np.hstack((self.max_exps, max_exp))

            if self.best_params is None:
                self.best_params = np.array(best_params)
            else:
                self.best_params = np.column_stack((self.best_params, best_params))

            best_ansatz = self.ansatz(best_params[0], best_params[1])
            self.best_ansatz = np.hstack((self.best_ansatz, best_ansatz))

            self.depth += 1

    def show_landscape(self, p, **plot_options):
        defaults = {
            'figsize': (16, 9),
        }
        defaults.update(plot_options)
        figsize = defaults['figsize']

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=figsize)

        if self.gmesh is None or self.bmesh is None or self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        exp_arr = self.exp_arr[:, :, p-1]
        surf = ax.plot_surface(self.gmesh, self.bmesh, exp_arr, cmap=cm.coolwarm)
        ax.set_xlabel('gamma')
        ax.set_ylabel('beta')
        ax.set_zlabel('expectation')
        fig.colorbar(surf, shrink=.5)

        plt.show()

    def show_heatmap(self, p):
        if self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        exp_arr = self.exp_arr[:, :, p-1]
        plt.xlabel('gamma_p/2pi')
        plt.ylabel('beta_p/pi')
        plt.imshow(exp_arr, cmap=cm.coolwarm, origin='lower', extent=[0, 1, 0, 1])

    def plot_alpha(self, alpha=True):
        p_range = range(1, self.depth + 1)

        fig, ax = plt.subplots()
        ax.set_xlabel('Circuit depth, $p$')
        ax.grid(True)
        ax.set_xticks(p_range)
        if alpha:
            ax.set_ylabel('Approx. ratio')
            ax.plot(p_range, self.max_exps/self.maxcut, marker='.')
        else:
            ax.set_ylabel('Expectation')
            ax.plot(p_range, self.max_exps, marker='.')

        plt.show()

    def status(self):
        if not self.depth:
            print('Paramters fixing not yet executed.')
        else:
            print(f'Parameters fixing executed up to p={self.depth}.')

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
