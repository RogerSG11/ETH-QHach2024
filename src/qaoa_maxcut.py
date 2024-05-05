import numpy as np
from typing import List, Type, Any
import networkx as nx

import cudaq
from cudaq import spin

'''
Prepare QAOA kernel based on graph

Args:
    - graph (nx.Graph)
'''
def build_kernel(graph):
    num_qubits = graph.number_of_nodes()
    # Get edge data as lists of basic types for kernel
    edges = graph.edges(data="weight")
    num_edges = len(edges)
    sources = []
    dests = []
    weights = []
    for x, y, w in edges:
        sources.append(x)
        dests.append(y)
        weights.append(w)

    # Kernel: it is very picky!
    @cudaq.kernel
    def kernel_qaoa(num_layers: int, thetas: List[float]):
        """QAOA ansatz for Max-Cut"""
        qvector = cudaq.qvector(num_qubits)
        # Create superposition
        h(qvector)
        # Loop over the layers
        for layer in range(num_layers):
            # Hamiltonian unitary
            for i in range(num_edges):
                src = sources[i]
                dst = dests[i]
                weight = weights[i]
                # RZZ gate = CNOT + RZ + CNOT
                # Ref: https://quantumcomputing.stackexchange.com/questions/30071/rzz-from-cnot-and-rz
                x.ctrl(qvector[src], qvector[dst])
                rz(2.0 * thetas[layer] * weight, qvector[dst])
                x.ctrl(qvector[src], qvector[dst])
            # Mixer unitary
            for qubit in range(num_qubits):
                rx(2.0 * thetas[layer + num_layers], qvector[qubit])
    
    return kernel_qaoa

'''
Create Ising hamiltonian for MaxCut using given edges defining a weighted graph.

Args:
    - graph (nx.Graph)

Ref:
    [1] https://qiskit-community.github.io/qiskit-optimization/tutorials/06_examples_max_cut_and_tsp.html
'''
def get_hamiltonian(graph):
    H = 0
    for x, y, w in graph.edges(data="weight"):
        H += spin.z(x)*spin.z(y)*w
    return H

'''
Run MaxCut on QAOA

Args:
    - num_qubits: number of qubits (i.e. number of nodes)
    - num_layers: number of layers for QAOA (arbitrary)
    - hamilatonian: hamiltonian (from get_hamiltonian)
'''
def run_qaoa_maxcut(graph, num_layers: int):
    # QAOA settings
    num_qubits = graph.number_of_nodes()
    num_parameters = 2*num_layers

    # Construct hamiltonian
    hamiltonian = get_hamiltonian(graph)
    print(hamiltonian)

    # Specify the optimizer and its initial parameters.
    #cudaq.set_random_seed(12)
    optimizer = cudaq.optimizers.COBYLA()
    #np.random.seed(12)
    optimizer.initial_parameters = np.random.uniform(-np.pi/8.0, np.pi/8.0, num_parameters)
    print("Initial parameters = ", optimizer.initial_parameters)

    # Build kernel from graph
    kernel_qaoa = build_kernel(graph)

    # Define the objective, return `<state(params) | H | state(params)>`
    def objective(parameters):
        return cudaq.observe(kernel_qaoa, hamiltonian, num_layers, parameters).expectation()
    # Optimize!
    optimal_expectation, optimal_parameters = optimizer.optimize(dimensions=num_parameters, function=objective)

    # Print the optimized value and its parameters
    print("Optimal value = ", optimal_expectation)
    print("Optimal parameters = ", optimal_parameters)

    # Sample the circuit using the optimized parameters
    counts = cudaq.sample(kernel_qaoa, num_layers, optimal_parameters, shots_count=10000)
    print(counts)
    print(counts.most_probable())

    return 0

if __name__ == "__main__":
    # Test
    edges = [(0,1,1), (0,2,2), (0,3,1), (1,2,1), (1,3,2), (2,3,1)]
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    run_qaoa_maxcut(graph, 8)
