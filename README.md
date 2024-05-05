# ETH-QHACK2024: NVIDIA CHALLENGE

## Team SovrapponiMi & Platyrinchos

Sources for the solution, separated by steps, are in `src`:
- Clustering: reduce CVRP to multiple instances of TSP by clustering the nodes
- Qubo 2 MaxCut: reduction of the TSP problem to MaxCut via QUBO formulation
- Graph shrinking: graph-size reduction of a MaxCut instance into smaller instances
- QAOA MaxCut: QAOA solver for MaxCut problem 
