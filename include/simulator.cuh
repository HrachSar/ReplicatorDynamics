// simulator.cuh
//
// GPU Monte-Carlo simulation of a replicator equation
//
//     dx/dt = x (1 - x) (beta + (alpha - beta) x + eta(t)),   x in (0, 1)
//
// driven by an external field eta(t) in {+eps, -eps} that is either
//   * STOCHASTIC    : random telegraph noise, switching at rate 2*rate,
//   * PERIODIC      : square wave of half-period 1/rate (random phase),
//   * DETERMINISTIC : eta = 0.
//
// Each of kNumTrajectories independent trajectories runs on one GPU thread.
// For every parameter value we estimate the probability that a trajectory
// reaches a threshold ("hit") and the mean hitting time among those that do.
//
// Build:  nvcc -std=c++17 -O2 src/simulator.cu main.cu -o replicator
#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int kNumTrajectories = 10000;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess)                                              \
            throw std::runtime_error(std::string(cudaGetErrorString(err_)) +  \
                                     " at " __FILE__ ":" +                    \
                                     std::to_string(__LINE__));               \
    } while (0)

enum class State { Stochastic, Periodic, Deterministic };

// Which quantity is swept in Simulator::RunSweep.
enum class Sweep {
    Position,  // initial condition x0 scanned across the interval between the
               // two unstable fixed points (x* for +eps and for -eps)
    Rate       // switching rate scanned over 0.1, 0.2, ...
};

struct SimulatorConfig {
    float alpha = 0.0f;
    float beta = 0.0f;
    float rate = 0.0f;
    float eps = 0.0f;
    float dt = 1e-3f;
    float tmin = 0.0f;
    float tmax = 0.0f;  // filled in by Simulator (see ComputeTmax)

    // A trajectory counts as a "hit" when x exceeds this value.
    float hit_threshold_stochastic = 0.77f;
    float hit_threshold_other = 0.99f;  // periodic and deterministic

    unsigned long long seed = 12345ULL;
};

struct OutputPaths {
    std::string deterministic = "results_deterministic.txt";
    std::string periodic = "results_periodic.txt";
    std::string stochastic = "hit_1.txt";
};

// Heuristic end-of-simulation time: 100 x the slowest characteristic time scale.
float ComputeTmax(const SimulatorConfig& cfg);

namespace stats {
// 95% Wilson score interval for a binomial proportion p_hat from n trials.
struct Interval { double lo, hi; };
Interval Wilson95(double p_hat, int n);
}  // namespace stats

struct CudaFree {
    void operator()(void* p) const noexcept { cudaFree(p); }
};
template <class T>
using DevPtr = std::unique_ptr<T, CudaFree>;

class Simulator {
public:
    Simulator(SimulatorConfig config, OutputPaths paths, int threads_per_block = 256);

    // Runs `state` for sweep indices j in [start, end) and writes one row per j to
    // the output file of that state (truncated on each call). Columns:
    //   param  mean_hit_time  hit_prob  hit_prob_se  wilson_lo  wilson_hi
    // `init_val` is used only for Sweep::Rate; values outside (0,1) mean
    // "uniform random initial condition".
    void RunSweep(State state, Sweep sweep, int start, int end, float init_val = 0.0f);

    // Runs the simulation once with the parameters in config() and writes one row
    // per trajectory (kNumTrajectories rows) to the state's output file, or to
    // `path_override` if given. Columns: trajectory  final_x  hit_time  hit
    // `final_x` is x at the end of the run, or at the hit if the trajectory hit;
    // `hit_time` is 0 when hit == 0. `init_val` outside (0,1) = random start.
    void RunTrajectories(State state, float init_val = 0.0f,
                         const std::string& path_override = "");

    const SimulatorConfig& config() const { return config_; }

private:
    const std::string& PathFor(State state) const;

    SimulatorConfig config_;
    OutputPaths paths_;
    int threads_;
    int blocks_;
    DevPtr<float> d_hit_time_;
    DevPtr<int> d_hit_;
    DevPtr<float> d_final_x_;
    std::vector<float> h_hit_time_;
    std::vector<int> h_hit_;
    std::vector<float> h_final_x_;
};
