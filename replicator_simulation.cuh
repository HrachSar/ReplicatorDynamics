#pragma once 
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

__global__ void SimulateDynamicsPeriodic(unsigned long long seed);
__global__ void SimulateDynamics(unsigned long long seed);
__global__ void SolveDynamicsDeterministic(unsigned long long seed);

class Simulator{

public:
    int m_blocks;
    int m_threads;
    float m_alpha;
    float m_beta;
    float m_eps;
    float m_rate;
    float m_dt;
    float m_tmin;
    float m_tmax;
    float A[2] = {0};
    std::vector<float> m_hresults;
    std::vector<float> m_htimes;
    std::string_view m_results_deterministic;
    std::string_view m_results_periodic;
    std::string_view m_results_stochastic;
    float *m_dresults;
    float *m_dtimes;
    std::fstream m_det_file;
    std::fstream m_stc_file;
    std::fstream m_per_file;


    Simulator(int blocks, int threads, float alpha, float beta, float rate, float eps, float dt, float tmin, float tmax, 
            std::string_view results_deterministic, std::string_view m_results_periodic, std::string_view results_stochastic);
    __host__ void AllocateMemory();
    __host__ void FreeMemory();
    static __device__ float Df(float x, float eps);
    static __device__ float MidpointSolverf(float x, float eps, float dt);
    static __device__ float Dg(float rate);
    static __device__ float MidpointSolverg(float rate, float h);
    __host__ void WriteIntoFiles(std::vector<float>& res, std::vector<float>& times, std::fstream& stream);
    __host__ float ComputeMean(std::vector<float>& res);
    __host__ void ComputeRateValsPeriodic(int start_rate, int end_rate);
    __host__ void ComputeRateValsStochastic(int start_rate, int end_rate);
    __host__ void ComputeRateValsDeterministic(int start_rate, int end_rate);

    template <typename... Args>
    static __host__ constexpr auto MyMax(Args&& ...args){
        return std::max({std::forward<Args>(args)...});
    }
    ~Simulator();
};


