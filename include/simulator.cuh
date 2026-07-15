#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <iostream>
#include <string_view>
#include <vector>
#include <fstream>


#define N  50000
__constant__ float d_A[2];
extern float A[2];

enum State{
    STOCHASTIC,
    PERIODIC,
    DETERMINISTIC
};

enum Sim_Type{
    POSITION,
    RATE
};

template <typename... Args>
static __host__ constexpr auto MyMax(Args&& ...args){
    return std::max({std::forward<Args>(args)...});
}

namespace Kernels{
    __global__ void SimulateDynamics(State state, unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times, float *hit, float init_val = 0, bool compute_times = false);
}

class SimulatorConfig{
    private:
        float m_alpha;
        float m_beta;
        float m_eps;
        float m_rate;
        float m_dt;
        float m_tmin;
        float m_tmax = 0.0f;
    public:
        SimulatorConfig(float alpha, float beta, float rate, 
                        float eps, float dt, float tmin);
        __host__ float GetAlpha() const;
        __host__ float GetBeta() const;
        __host__ float GetEps() const;
        __host__ float GetRate() const;
        __host__ float GetDt() const;
        __host__ float GetTmin() const;
        __host__ float GetTmax() const;
        __host__ void SetAlpha(float alpha);
        __host__ void SetBeta(float beta);
        __host__ void SetEps(float eps);
        __host__ void SetRate(float rate);
        __host__ void SetDt(float dt);
        __host__ void SetTmin(float tmin);
        __host__ void SetTmax(float tmax);
        ~SimulatorConfig();
};

class PathManager{

    public:
        std::fstream m_det_file;
        std::fstream m_stc_file;
        std::fstream m_per_file;
        __host__ std::string_view GetDetPath() const;
        __host__ std::string_view GetStcPath() const;
        __host__ std::string_view GetPerPath() const;
        __host__ void SetDetPath(std::string_view path);
        __host__ void SetStcPath(std::string_view path);
        __host__ void SetPerPath(std::string_view path);
        PathManager(std::string_view results_deterministic, std::string_view results_periodic, std::string_view results_stochastic);
        PathManager(PathManager&& other) noexcept;
        __host__ void WriteIntoFiles(std::vector<float>& res, std::vector<float>& times, std::vector<float>& hits, std::fstream& stream);
        ~PathManager();
    private:
        std::string_view m_results_deterministic;
        std::string_view m_results_periodic;
        std::string_view m_results_stochastic;
};

class ResourceManager{

    public:
        std::vector<float> m_hresults;
        std::vector<float> m_htimes;
        std::vector<float> m_hhits;

        float *m_dresults;
        float *m_dtimes;
        float *m_dhits;
        __host__ void AllocateMemory();
        __host__ void FreeMemory();
        ResourceManager::ResourceManager(ResourceManager&& other) noexcept;
        ResourceManager();
        ~ResourceManager();
};

class Simulator{

    public:
        SimulatorConfig m_config;
        PathManager m_path_manager;
        ResourceManager m_resource_manager;
        static __device__ float Df(float x, float eps);
        static __device__ float MidpointSolverf(float x, float eps, float dt);
        static __device__ float Dg(float rate);
        static __device__ float MidpointSolverg(float rate, float h);
        __host__ float ComputeMean(std::vector<float>& res);
        __host__ void ComputeRateVals(State state, Sim_Type type, int start_rate, int end_rate, float x = 0);
        Simulator(int blocks, int threads, SimulatorConfig& config, PathManager& path_manager, ResourceManager& resource_manager);
        ~Simulator();
        __host__ int GetNumBlocks();
        __host__ int GetNumThreads();
        __host__ void SetGridSize(int size);
        __host__ void SetBlockSize(int size);
    private:
        int m_blocks;
        int m_threads;
};

#endif