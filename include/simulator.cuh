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


#define N  10000
__constant__ float d_A[2];
extern float A[2];

template <typename... Args>
static __host__ constexpr auto MyMax(Args&& ...args){
    return std::max({std::forward<Args>(args)...});
}

namespace Kernels{
    __global__ void SimulateDynamics(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times);
    __global__ void SolveDynamicsDeterministic(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times);
    __global__ void SimulateDynamicsPeriodic(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times);
};

class SimulatorConfig{
    private:
        float m_alpha;
        float m_beta;
        float m_eps;
        float m_rate;
        float m_dt;
        float m_tmin;
        float m_tmax;
    public:
        SimulatorConfig(float alpha, float beta, float rate, 
                        float eps, float dt, float tmin, float tmax);
        __host__ float GetAlpha();
        __host__ float GetBeta();
        __host__ float GetEps();
        __host__ float GetRate();
        __host__ float GetDt();
        __host__ float GetTmin();
        __host__ float GetTmax();
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
        __host__ std::string_view GetDetPath();
        __host__ std::string_view GetStcPath();
        __host__ std::string_view GetPerPath();
        __host__ void SetDetPath(std::string_view path);
        __host__ void SetStcPath(std::string_view path);
        __host__ void SetPerPath(std::string_view path);
        PathManager(std::string_view results_deterministic, std::string_view results_periodic, std::string_view results_stochastic);
        PathManager(PathManager&& other) noexcept;
        __host__ void WriteIntoFiles(std::vector<float>& res, std::vector<float>& times, std::fstream& stream);
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
        float *m_dresults;
        float *m_dtimes;
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
        __host__ void ComputeRateValsPeriodic(int start_rate, int end_rate);
        __host__ void ComputeRateValsStochastic(int start_rate, int end_rate);
        __host__ void ComputeRateValsDeterministic(int start_rate, int end_rate);
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