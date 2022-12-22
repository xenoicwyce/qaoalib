# qaoalib
A package for QAOA Max-cut Calculations.

Packages required:
- numpy
- networkx
- matplotlib
- qiskit

# How to install
You can install from the PyPI:
```
pip install --upgrade qaoalib
```

# Usage
Calculate Max-cut expectation with `Qmc` or `QmcFastKron` (faster version):
```
import networkx as nx
from qaoalib.qaoa import Qmc, QmcFastKron

G = nx.fast_gnp_random_graph(10, 0.5)  # Random graph with 10 nodes
params = [0.2, 0.4, 0.3, 0.5]          # 4 params, depth = 2

qmc = Qmc(G, params) # or QmcFastKron(G, params)
qmc.run()
print(qmc.expectation)
```

Plot landscapes of the QAOA Max-cut expectation function:
```
import networkx as nx
from qaoalib.qaoa.landscape import HybridFast

G = nx.fast_gnp_random_graph(10, 0.5)
prev_params = [0.1, 0.2] # previous parameters (gamma1, beta1)

ins = HybridFast(G, prev_params) # plots the landscape wrt gamma2 and beta2 with previous parameters given.
ins.create_grid()
ins.show_landscape()
```
