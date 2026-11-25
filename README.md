# GPU Replicator Dynamics under a Switching Field

CUDA Monte-Carlo simulation of a one-dimensional replicator equation driven by a dichotomous noise that switches between two values. Thousands of independent
trajectories run in parallel (one per GPU thread), and the program estimates how
likely, and how quickly, the system escapes to a target state.

## Model

The state `x` in (0, 1) obeys

```
dx/dt = x (1 - x) ( beta + (alpha - beta) x + eta(t) )
```

with `eta(t)` taking the values `+eps` or `-eps`. For each fixed value of `eta` the
dynamics are bistable: `x = 0` and `x = 1` are stable, and an interior fixed point


| Mode            | Field `eta(t)`                                                                    |
|-----------------|-----------------------------------------------------------------------------------|
| `Stochastic`    | Random telegraph noise: sign chosen at random, segment lengths `Exp(2*rate)`      |
| `Periodic`      | Square wave `+eps / -eps`, half-period `1/rate`, randomised end phase             |
| `Deterministic` | `eta = 0`                                                                         |

A trajectory **hits** when `x` exceeds a threshold (0.77 for `Stochastic`, 0.99 for
`Periodic` and `Deterministic`; see `SimulatorConfig`). Integration uses classical
RK4 with step `dt`; the last step of each constant-field segment is shortened so
segment lengths are exact.

## Requirements

- NVIDIA GPU and the CUDA Toolkit (`nvcc`, cuRAND)
- A C++17 host compiler

## Build and run

```bash
nvcc -std=c++17 -O2 src/simulator.cu main.cu -o replicator
./replicator
```

## Project layout

```
include/simulator.cuh   public interface: config, Simulator class, helpers
src/simulator.cu        CUDA kernel, host logic, statistics, file output
main.cu                 example driver
```

## Usage

Parameters live in `SimulatorConfig`; output files in `OutputPaths`.

```cpp
Simulator sim(config, OutputPaths{}, /*threads_per_block=*/256);

// (a) Parameter sweep: one statistics row per sweep point
sim.RunSweep(State::Stochastic, Sweep::Position, 0, 100);

// (b) Single run: one row per trajectory
sim.RunTrajectories(State::Stochastic, 0.0f, "trajectories_stochastic.txt");
```

`tmax` is computed automatically by `ComputeTmax` as 100 times the slowest
characteristic time scale of the system (about 250 for the defaults above).

### Sweeps (`RunSweep`)

`RunSweep(state, sweep, start, end, init_val)` runs `kNumTrajectories` trajectories
for each sweep index `j` in `[start, end)`.

| `Sweep`    | Parameter varied                                                                                         |
|------------|----------------------------------------------------------------------------------------------------------|
| `Position` | Initial condition, from just above `x*(+eps)` towards `x*(-eps)`, in steps of 1% of that interval        |
| `Rate`     | Switching rate `0.1, 0.2, ...` (`(j + 1) * 0.1`); initial condition `init_val`, or random if not in (0,1) |

Output (tab-separated, with a header row):

| Column          | Meaning                                                              |
|-----------------|----------------------------------------------------------------------|
| `param`         | Swept value (initial `x` or rate)                                    |
| `mean_hit_time` | Mean hitting time over trajectories that hit (`nan` if none did)     |
| `hit_prob`      | Fraction of trajectories that hit                                    |
| `hit_prob_se`   | Binomial standard error `sqrt(p(1-p)/N)`                             |
| `wilson_lo/hi`  | 95% Wilson score interval for `hit_prob`                             |

### Single runs (`RunTrajectories`)

Runs all trajectories once with the configured parameters and writes one row each:

| Column       | Meaning                                                                   |
|--------------|---------------------------------------------------------------------------|
| `trajectory` | Trajectory index                                                          |
| `final_x`    | `x` at the end of the run, or at the moment of the hit                    |
| `hit_time`   | Time of the hit (0 if the trajectory did not hit)                         |
| `hit`        | 1 if the trajectory hit, else 0                                           |

## Notes and limitations

- **Number of trajectories** is the compile-time constant `kNumTrajectories`
  (10 000) in `simulator.cuh`.
- **Errors.** CUDA failures and file errors throw `std::runtime_error`; `main`
  prints the message and exits with status 1.
- **Memory.** Device buffers are owned by `std::unique_ptr` with a `cudaFree`
  deleter (`DevPtr`), so they are released automatically. Keep `Simulator` as a
  local object, not a global or `static`, so it is destroyed before the CUDA
  context shuts down.