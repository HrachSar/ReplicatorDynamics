// simulator.cu
#include "../include/simulator.cuh"

#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

namespace {

// Everything the kernel needs, passed by value (replaces the old global
// constant-memory array, so several Simulators can coexist safely).
struct KernelArgs {
    State state;
    unsigned long long seed;
    float alpha, beta, eps, rate, dt, tmin, tmax;
    float init_val;   // in (0,1): fixed start; otherwise uniform random start
    float threshold;  // hit when x > threshold
};

__device__ __forceinline__ float F(float x, const KernelArgs& a, float eta) {
    return x * (1.0f - x) * (a.beta + (a.alpha - a.beta) * x + eta);
}

// One classical RK4 step of size h with constant field eta.
__device__ __forceinline__ float RK4Increment(float x, float h, const KernelArgs& a, float eta) {
    const float k1 = F(x, a, eta);
    const float k2 = F(x + 0.5f * h * k1, a, eta);
    const float k3 = F(x + 0.5f * h * k2, a, eta);
    const float k4 = F(x + h * k3, a, eta);
    return h * (k1 + 2.0f * (k2 + k3) + k4) / 6.0f;
}

// Integrates for `duration` with constant field `eta`, stopping early on a hit.
// The last step is shortened so segments have their exact length.
// Returns true on hit; x and t are updated in place.
__device__ bool Advance(float& x, float& t, float duration, float eta, const KernelArgs& a) {
    const float t_end = t + duration;
    while (t < t_end) {
        const float h = fminf(a.dt, t_end - t);
        x += RK4Increment(x, h, a, eta);
        t += h;
        if (x > a.threshold) return true;
    }
    return false;
}

__global__ void SimulateKernel(KernelArgs a, float* hit_time, int* hit, float* final_x) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= kNumTrajectories) return;

    curandState rng;
    curand_init(a.seed, id, 0, &rng);

    float x = (a.init_val > 0.0f && a.init_val < 1.0f) ? a.init_val : curand_uniform(&rng);
    float t = a.tmin;
    bool hit_flag = false;

    switch (a.state) {
        case State::Stochastic: {
            // Field flips sign at Poisson times of rate 2*rate; the sign of each
            // segment is chosen uniformly at random.
            while (t < a.tmax && !hit_flag) {
                const float dwell = -logf(curand_uniform(&rng)) / (2.0f * a.rate);
                const float eta = (curand_uniform(&rng) > 0.5f) ? a.eps : -a.eps;
                hit_flag = Advance(x, t, dwell, eta, a);
            }
            break;
        }
        case State::Periodic: {
            // Square wave of half-period 1/rate. The end time is extended by a
            // random fraction of a full period to randomise the final phase.
            const float half_period = 1.0f / a.rate;
            const float t_final = a.tmax + curand_uniform(&rng) * (2.0f / a.rate);
            float eta = a.eps;
            while (t < t_final && !hit_flag) {
                eta = -eta;
                hit_flag = Advance(x, t, fminf(half_period, t_final - t), eta, a);
            }
            break;
        }
        case State::Deterministic: {
            hit_flag = Advance(x, t, a.tmax - a.tmin, 0.0f, a);
            break;
        }
    }

    hit[id] = hit_flag ? 1 : 0;
    hit_time[id] = hit_flag ? t : 0.0f;
    final_x[id] = x;
}

template <class T>
DevPtr<T> DeviceAlloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return DevPtr<T>(p);
}

}  // namespace

float ComputeTmax(const SimulatorConfig& c) {
    const float a = c.alpha, b = c.beta, e = c.eps;
    const float t = 100.0f * std::max({std::fabs(1.0f / (a + e)),
                                       std::fabs(1.0f / (a - e)),
                                       std::fabs(1.0f / (b + e)),
                                       std::fabs(1.0f / (b - e)),
                                       std::fabs(a / ((a - e) * (b - e))),
                                       std::fabs(a / ((a + e) * (a + e)))});
    if (!std::isfinite(t))
        throw std::invalid_argument("ComputeTmax: degenerate parameters (division by zero)");
    return t;
}

stats::Interval stats::Wilson95(double p, int n) {
    const double z = 1.959964, z2 = z * z, nn = n;
    const double denom = 1.0 + z2 / nn;
    const double centre = (p + z2 / (2.0 * nn)) / denom;
    const double half = z / (2.0 * nn) * std::sqrt(4.0 * nn * p * (1.0 - p) + z2) / denom;
    return {centre - half, centre + half};
}

Simulator::Simulator(SimulatorConfig config, OutputPaths paths, int threads_per_block)
    : config_(config),
      paths_(std::move(paths)),
      threads_(threads_per_block),
      blocks_((kNumTrajectories + threads_per_block - 1) / threads_per_block),
      d_hit_time_(DeviceAlloc<float>(kNumTrajectories)),
      d_hit_(DeviceAlloc<int>(kNumTrajectories)),
      d_final_x_(DeviceAlloc<float>(kNumTrajectories)),
      h_hit_time_(kNumTrajectories, 0.0f),
      h_hit_(kNumTrajectories, 0),
      h_final_x_(kNumTrajectories, 0.0f) {
    config_.tmax = ComputeTmax(config_);
}

const std::string& Simulator::PathFor(State s) const {
    switch (s) {
        case State::Stochastic: return paths_.stochastic;
        case State::Periodic: return paths_.periodic;
        case State::Deterministic: return paths_.deterministic;
    }
    throw std::logic_error("unknown State");
}

void Simulator::RunSweep(State state, Sweep sweep, int start, int end, float init_val) {
    std::ofstream out(PathFor(state), std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("Failed to open output file: " + PathFor(state));
    out << std::setprecision(8)
        << "param\tmean_hit_time\thit_prob\thit_prob_se\twilson_lo\twilson_hi\n";

    const SimulatorConfig& c = config_;
    const double span = static_cast<double>(c.beta) - c.alpha;
    // Unstable fixed points of the dynamics under eta = +eps and eta = -eps.
    const double x_lo = (c.beta + c.eps) / span;
    const double x_hi = (c.beta - c.eps) / span;

    for (int j = start; j < end; ++j) {
        KernelArgs args{state, c.seed, c.alpha, c.beta, c.eps, c.rate,
                        c.dt, c.tmin, c.tmax, init_val,
                        state == State::Stochastic ? c.hit_threshold_stochastic
                                                   : c.hit_threshold_other};
        float param = 0.0f;
        if (sweep == Sweep::Rate) {
            args.rate = (j + 1) * 0.1f;
            param = args.rate;
        } else {
            const double x_start = x_lo + 0.001;
            args.init_val = static_cast<float>(x_start + (x_hi - x_start) * (0.01 * j));
            param = args.init_val;
        }

        SimulateKernel<<<blocks_, threads_>>>(args, d_hit_time_.get(), d_hit_.get(),
                                             d_final_x_.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_hit_time_.data(), d_hit_time_.get(),
                              kNumTrajectories * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_hit_.data(), d_hit_.get(),
                              kNumTrajectories * sizeof(int), cudaMemcpyDeviceToHost));

        double time_sum = 0.0;
        long long n_hits = 0;
        for (int i = 0; i < kNumTrajectories; ++i) {
            n_hits += h_hit_[i];
            time_sum += h_hit_time_[i];
        }
        const double p = static_cast<double>(n_hits) / kNumTrajectories;
        const double se = std::sqrt(p * (1.0 - p) / kNumTrajectories);
        // Mean hitting time is conditioned on having hit (NaN if no trajectory did).
        const double mean_time =
            n_hits > 0 ? time_sum / n_hits : std::numeric_limits<double>::quiet_NaN();
        const stats::Interval w = stats::Wilson95(p, kNumTrajectories);

        out << param << '\t' << mean_time << '\t' << p << '\t' << se << '\t' << w.lo << '\t'
            << w.hi << '\n';
        out.flush();
    }
}

void Simulator::RunTrajectories(State state, float init_val, const std::string& path_override) {
    const std::string& path = path_override.empty() ? PathFor(state) : path_override;
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("Failed to open output file: " + path);

    const SimulatorConfig& c = config_;
    const KernelArgs args{state, c.seed, c.alpha, c.beta, c.eps, c.rate,
                          c.dt, c.tmin, c.tmax, init_val,
                          state == State::Stochastic ? c.hit_threshold_stochastic
                                                     : c.hit_threshold_other};

    SimulateKernel<<<blocks_, threads_>>>(args, d_hit_time_.get(), d_hit_.get(),
                                         d_final_x_.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_hit_time_.data(), d_hit_time_.get(),
                          kNumTrajectories * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hit_.data(), d_hit_.get(),
                          kNumTrajectories * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_final_x_.data(), d_final_x_.get(),
                          kNumTrajectories * sizeof(float), cudaMemcpyDeviceToHost));

    out << std::setprecision(8) << "trajectory\tfinal_x\thit_time\thit\n";
    for (int i = 0; i < kNumTrajectories; ++i)
        out << i << '\t' << h_final_x_[i] << '\t' << h_hit_time_[i] << '\t' << h_hit_[i] << '\n';
}