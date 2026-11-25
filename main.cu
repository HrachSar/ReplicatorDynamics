// main.cu
#include <cstdio>
#include <exception>

#include "include/simulator.cuh"

int main() {
    try {
        SimulatorConfig config;
        config.alpha = 0.6f;
        config.beta = -2.0f * config.alpha;
        config.eps = 0.2f;
        config.rate = 0.1f;
        config.dt = 1e-3f;
        config.tmin = 0.0f;

        OutputPaths paths;  // defaults: results_deterministic.txt, results_periodic.txt, hit_1.txt

        Simulator sim(config, paths, /*threads_per_block=*/256);
        std::printf("Tmax = %f\n", sim.config().tmax);

        // Sweep the initial position across the bistable interval (100 points),
        // stochastic field.
        sim.RunSweep(State::Stochastic, Sweep::Position, 0, 100);

        // Single run of kNumTrajectories trajectories at the fixed parameters above,
        // one output row per trajectory (random initial conditions).
        sim.RunTrajectories(State::Stochastic, 0.0f, "trajectories_stochastic.txt");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
