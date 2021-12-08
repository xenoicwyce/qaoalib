import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm

DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-8

class QmcLandscapeBase:
    """
    This is a base class.
    It should be only used for inhertiance.
    """
    def __init__(self, G, prev_params=None):
        if prev_params is not None:
            if len(prev_params) % 2:
                raise ValueError(f'Constructor failed. prev_params must have even length, got {len(prev_params)}.')
        self.graph = G
        self.num_qubits = len(G.nodes)
        self.edge_list = list(G.edges)
        self.prev_params = prev_params
        self.gmesh = None
        self.bmesh = None
        self.exp_arr = None
        self.depth = 1 if prev_params is None else len(prev_params)//2+1
        self.npts = None

    def get_max(self, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        if self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        exp_max = np.max(self.exp_arr)
        whr = np.where(np.isclose(self.exp_arr, exp_max, rtol=rtol, atol=atol))
        indices = zip(whr[0], whr[1])
        angle_list = [(self.gmesh[idx], self.bmesh[idx]) for idx in indices]
        return (exp_max, angle_list)

    def get_min(self, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        if self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        exp_min = np.min(self.exp_arr)
        whr = np.where(np.isclose(self.exp_arr, exp_min, rtol=rtol, atol=atol))
        indices = zip(whr[0], whr[1])
        angle_list = [(self.gmesh[idx], self.bmesh[idx]) for idx in indices]
        return (exp_min, angle_list)

    def show_landscape(self, **plot_options):
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

    def show_heatmap(self):
        if self.exp_arr is None:
            raise ValueError('Grid not found. Run create_grid() method first.')

        plt.xlabel('gamma_p/2pi')
        plt.ylabel('beta_p/pi')
        plt.imshow(self.exp_arr, cmap=cm.coolwarm, origin='lower', extent=[0, 1, 0, 1])
