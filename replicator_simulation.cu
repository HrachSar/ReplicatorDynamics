#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <iostream>

#include "include/simulator.cuh"

int main(){

	float a = 0.3f;
	float eps = 0.2f;
	float rate = 0.1f;
	float h = 0.01;
	float tmin = 0;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
	float tmax = 1000.0f * MyMax(std::abs(1.0f / (2*a + eps)), std::abs(1.0f / (2*a - eps)), std::abs(1.0f / (2*a + eps)), std::abs(1.0f / (2*a - eps)), 
                            std::abs(2*a / ((2*a - eps)*(2*a - eps))), std::abs(2*a / ((2*a + eps)*(2*a + eps))));
    
    std::string_view path_deterministic = "results_deterministic.txt";
    std::string_view path_stochastic = "results.txt";
    std::string_view path_periodic = "results_periodic.txt";
    
    SimulatorConfig config(2*a, 2*a, rate, eps, h, tmin, tmax);
    PathManager path_manager(path_deterministic, path_periodic, path_stochastic);
    ResourceManager res = ResourceManager();

    Simulator sim(blocks, threads, config, path_manager, res);
    printf("Tmax = %f\n", tmax);

    // Kernels::SimulateDynamics<<<sim.GetNumBlocks(), sim.GetNumThreads()>>>(12345ULL, sim.m_config.GetTmin(), sim.m_config.GetTmax(), 
    //                      sim.m_config.GetEps(), sim.m_config.GetRate(), sim.m_config.GetDt(), sim.m_resource_manager.m_dresults, sim.m_resource_manager.m_dtimes);
    // cudaDeviceSynchronize();
    // cudaMemcpy(sim.m_resource_manager.m_hresults.data(), sim.m_resource_manager.m_dresults, N * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(sim.m_resource_manager.m_htimes.data(), sim.m_resource_manager.m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost); 

    // sim.m_path_manager.WriteIntoFiles(sim.m_resource_manager.m_hresults, sim.m_resource_manager.m_htimes, sim.m_path_manager.m_stc_file);
    
    // simulator.AllocateMemory();
    sim.ComputeRateValsPeriodic(0, 100);
    cudaDeviceSynchronize();
    // simulator.FreeMemory();
    
    return 0;
}            