# qaoalib
Implementations of VQA simulations for combinatorial optimization (mainly with graph and Max-cut).

v0.2 works towards adapting the circuit simulations with new Qiskit primitives: `Sampler` and `Estimator` (starting from Qiskit 1.0).

# How to install
You can install from the PyPI:
```
pip install --upgrade qaoalib
```

# Usage
Solving Max-cut with VQE:
```
from qaoalib.solver import VQE
from qiskit_optimization.applications import Maxcut

G = nx.random_regular_graph(3, 6) # 3-regular graph wtih 6 nodes
qp = Maxcut(G).to_quadratic_program()
ansatz = # some preferred ansatz
solver = VQE(qp, ansatz)
result = solver.solve()
print(result)
```

# Usage (legacy)
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
