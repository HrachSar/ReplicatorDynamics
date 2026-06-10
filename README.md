## Stochastic Replicator Dynamics Simulator in CUDA

This project implements a simulator for stochastic replicator dynamics using CUDA. The underlying model is formulated as a Piecewise Deterministic Markov Process (PDMP), where environmental fluctuations are represented by a randomly switching state variable.

The simulation is based on the Gillespie algorithm for generating stochastic switching times, while the deterministic replicator dynamics between switching events are integrated using a second-order midpoint method.

### Features

- CUDA-accelerated parallel simulation of large populations of trajectories.
- Piecewise Deterministic Markov Process (PDMP) framework.
- Random telegraph noise generated via Gillespie sampling.
- Periodically fluctuating environment implementation.
- Computation of stationary probability distributions.
- Estimation of hitting probabilities and mean first-passage times.
- Comparison between deterministic, stochastic, and periodically forced environments.

### Requirements

- CUDA Toolkit 13.1
- NVCC Compiler
- C++17 compatible compiler
