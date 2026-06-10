#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <iostream>

#include "replicator_simulation.cuh"

__global__ void SimulateDynamics(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= N)
        return;

    curandState rng;
    curand_init(seed, id, 0, &rng);  

 	float x = curand_uniform(&rng);
	float X_temp = x;
    float t = tmin;
	while(t < tmax){

		float u  = curand_uniform(&rng);
		float dg_step = 0;
        float t0 = -logf(u + 1e-9f);
		float etta_temp = (curand_uniform(&rng) > 0.5f) ? eps : -eps;

		while(dg_step < t0){
			float df_step = Simulator::MidpointSolverf(X_temp, etta_temp, dt);
			dg_step += 2 * rate * dt;
			X_temp += df_step;
            if(X_temp < 0.99){
                times[id] = t + dg_step / (2 * rate);
            }
		}
		t += t0 / (2 * rate);

	}
    results[id] = X_temp;
}
__global__ void SimulateDynamicsPeriodic(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= N)
        return;

    curandState rng;
    curand_init(seed, id, 0, &rng);  

 	float x = curand_uniform(&rng);
	float X_temp = x;
    float u = curand_uniform(&rng);
    float half_period = 1.0f / rate;
    float etta_temp = eps;         
    float t = tmin;

    const float t_final = tmax + u * (2.0f / rate);
    
    while(t < t_final){
        etta_temp = -etta_temp;

        float t_next = t + half_period;
        if (t_next > t_final)
            t_next = t_final;

        while(t < t_next){
            float df_step = Simulator::MidpointSolverf(X_temp, etta_temp, dt);
            t += dt;
            X_temp += df_step;
            if(X_temp < 0.99){
                times[id] = t;
            }
        }
    }
    results[id] = X_temp;
}
__global__ void SolveDynamicsDeterministic(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= N)
        return;

    curandState rng;
    curand_init(seed, id, 0, &rng);  

 	float x = curand_uniform(&rng);
	float X_temp = x;
    float t = tmin;

    while(t < tmax){

        float df_step = Simulator::MidpointSolverf(X_temp, 0, dt);
        X_temp += df_step;

        if(X_temp < 0.99){
            times[id] = t;
        }
        t += dt;
    }
    results[id] = X_temp;
}

__host__ void Simulator::AllocateMemory(){

    cudaError_t err = cudaMalloc((void **)&(this->m_dresults), N * sizeof(float));
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&(this->m_dtimes), N * sizeof(float));
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    cudaMemcpyToSymbol(d_A, A, 2 * sizeof(float)); //for constant memory

    m_hresults.resize(N, 0);
    m_htimes.resize(N, 0);

}

__host__ void Simulator::FreeMemory(){
    cudaFree(m_dresults);
    cudaFree(m_dtimes);
    m_hresults.clear();
    m_htimes.clear();
}

__device__ float Simulator::Df(float x, float eps){
    float alpha = d_A[0];
	float beta = d_A[1];
	float dx = x * (1 - x) * (beta + (alpha - beta) * x + eps);

	return dx;
}

__device__ float Simulator::MidpointSolverf(float x, float eps, float dt){
	float Xmid = x;
	float r1 = Df(Xmid, eps);
	Xmid += dt * r1 * 0.5;
	float r2 = Df(Xmid, eps);
	r2 *= dt;

	return r2;
}

__device__ float Simulator::Dg(float rate){
	float dx = 2 * rate;

	return dx;
}

__device__ float Simulator::MidpointSolverg(float rate, float h){
	float r1 = h * Dg(rate);
	float r2 = h * Dg(rate + r1 * 0.5);

	return r2;
}

Simulator::Simulator(int blocks, int threads, float alpha, float beta, float rate, float eps, float dt, float tmin, float tmax, 
        std::string_view results_deterministic, std::string_view results_periodic, std::string_view results_stochastic){
    this->m_blocks = blocks;    
    this->m_threads = threads;
    this->m_alpha = alpha;
    this->m_beta = beta;
    this->m_rate = rate;
    this->m_eps = eps;
    this->m_dt = dt;
    this->m_tmin = tmin;
    this->m_tmax = tmax;
    this->m_results_deterministic = results_deterministic;
    this->m_results_periodic = results_periodic;
    this->m_results_stochastic = results_stochastic;
    this->m_det_file.open(m_results_deterministic.data(), std::ios::out | std::ios::trunc);
    this->m_stc_file.open(m_results_stochastic.data(), std::ios::out | std::ios::trunc);
    this->m_per_file.open(m_results_periodic.data(), std::ios::out | std::ios::trunc);

    if(!m_det_file || !m_per_file || !m_stc_file)
            throw std::runtime_error("Failed to open output files.");
    
    A[0] = m_alpha;
    A[1] = m_beta;
    AllocateMemory();
}


__host__ void Simulator::WriteIntoFiles(std::vector<float>& res, std::vector<float>& times, std::fstream& stream){
    if(!stream.is_open())
        throw std::exception("File stream is not open.");

    for(int i = 0; i < res.size(); i++){
        stream << res[i] << "\t" << times[i] << "\n";
    }

    stream.flush();
}

__host__ float Simulator::ComputeMean(std::vector<float>& res){
    float total = 0.0f;

    for(auto iter = res.cbegin(); iter != res.cend(); iter++){
        total += *(iter);
    }
    return (total / res.size());
}

__host__ void Simulator::ComputeRateValsStochastic(int start, int end){
    int j = start;
    std::vector<float> res = std::vector<float>(end - start, 0.0f);
    std::vector<float> time = std::vector<float>(end - start, 0.0f);

    for(;j < end; j++){
        cudaError_t err = cudaMemset(m_dtimes, 0, N * sizeof(float));
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 
        float i = (j + 1) * 0.1f;
        SimulateDynamics<<<m_blocks, m_threads>>>(12345ULL, m_tmin, m_tmax, m_eps, i, m_dt, m_dresults, m_dtimes);
        cudaDeviceSynchronize();

        err = cudaMemcpy(m_htimes.data(), m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 

        res[j - start] = i;
        time[j - start] = ComputeMean(m_htimes);
    }
    WriteIntoFiles(res, time, m_stc_file);
}

__host__ void Simulator::ComputeRateValsPeriodic(int start, int end){
    int j = start;
    std::vector<float> res = std::vector<float>(end - start, 0.0f);
    std::vector<float> time = std::vector<float>(end - start, 0.0f);

    for(;j < end; j++){
        cudaError_t err = cudaMemset(m_dtimes, 0, N * sizeof(float));
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 
        float i = (j + 1) * 0.1f;
        SimulateDynamicsPeriodic<<<m_blocks, m_threads>>>(12345ULL, m_tmin, m_tmax, m_eps, i, m_dt, m_dresults, m_dtimes);
        cudaDeviceSynchronize();

        err = cudaMemcpy(m_htimes.data(), m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 

        res[j - start] = i;
        time[j - start] = ComputeMean(m_htimes);
    }
    WriteIntoFiles(res, time, m_per_file);
}

__host__ void Simulator::ComputeRateValsDeterministic(int start, int end){
    int j = start;
    std::vector<float> res = std::vector<float>(end - start, 0.0f);
    std::vector<float> time = std::vector<float>(end - start, 0.0f);

    for(;j < end; j++){
        cudaError_t err = cudaMemset(m_dtimes, 0, N * sizeof(float));
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 
        float i = (j + 1) * 0.1f;
        SolveDynamicsDeterministic<<<m_blocks, m_threads>>>(12345ULL, m_tmin, m_tmax, m_eps, i, m_dt, m_dresults, m_dtimes);
        cudaDeviceSynchronize();

        err = cudaMemcpy(m_htimes.data(), m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 

        res[j - start] = i;
        time[j - start] = ComputeMean(m_htimes);
    }
    WriteIntoFiles(res, time, m_det_file);
}

Simulator::~Simulator(){
    FreeMemory();
    m_det_file.close();
    m_stc_file.close();
    m_per_file.close();
}

int main(){

	float a = 3.0f;
	float eps = 2.0f;
	float rate = 3.0f;
	float h = 0.01;
	float tmin = 0;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
	float tmax = 1000.0f * Simulator::MyMax(std::abs(1.0f / (-2*a + eps)), std::abs(1.0f / (-2*a - eps)), std::abs(1.0f / (2*a + eps)), std::abs(1.0f / (2*a - eps)), 
                            std::abs(-2*a / ((-2*a - eps)*(2*a - eps))), std::abs(-2*a / ((-2*a + eps)*(2*a + eps))));
    std::string_view path_deterministic = "results_deterministic.txt";
    std::string_view path_stochastic = "results.txt";
    std::string_view path_periodic = "results_periodic.txt";

    Simulator simulator(blocks, threads, -2*a, 2*a, rate, eps, h, tmin, tmax, path_deterministic, path_periodic, path_stochastic);
    printf("Tmax = %f\n", tmax);

    //simulator.ComputeRateValsPeriodic(0, 100);
    //cudaDeviceSynchronize();
    //simulator.FreeMemory();
    simulator.AllocateMemory();
    SimulateDynamics<<<simulator.m_blocks, simulator.m_threads>>>(12345ULL, simulator.m_tmin, simulator.m_tmax, 
                        simulator.m_eps, simulator.m_rate, simulator.m_dt, simulator.m_dresults, simulator.m_dtimes);
    cudaMemcpy(simulator.m_hresults.data(), simulator.m_dresults, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(simulator.m_htimes.data(), simulator.m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost); 

    cudaDeviceSynchronize();
    simulator.WriteIntoFiles(simulator.m_hresults, simulator.m_htimes, simulator.m_stc_file);
    simulator.FreeMemory();

    /*
    simulator.AllocateMemory();
    simulator.ComputeRateValsStochastic(0, 100);
    cudaDeviceSynchronize();
    simulator.FreeMemory();
    */
    return 0;
}