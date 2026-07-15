#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <iostream>

#include "include/simulator.cuh"

int main(){

	float a = 1.0f;
	float eps = 1.1f;
	float rate = 10.0f;
	float h = 0.001;
	float tmin = 0;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    std::string_view path_deterministic = "results_deterministic.txt";
    std::string_view path_stochastic = "results.txt";
    std::string_view path_periodic = "results_periodic.txt";
    
    SimulatorConfig config(2 * a, a, rate, eps, h, tmin);
    PathManager path_manager(path_deterministic, path_periodic, path_stochastic);
    ResourceManager res = ResourceManager();

    Simulator sim(blocks, threads, config, path_manager, res);
    printf("Tmax = %f\n", sim.m_config.GetTmax());

    Kernels::SimulateDynamics<<<sim.GetNumBlocks(), sim.GetNumThreads()>>>(STOCHASTIC, 12345ULL, sim.m_config.GetTmin(), sim.m_config.GetTmax(), 
                         sim.m_config.GetEps(), sim.m_config.GetRate(), sim.m_config.GetDt(), sim.m_resource_manager.m_dresults, sim.m_resource_manager.m_dtimes, 0, false);
    cudaDeviceSynchronize();
    cudaMemcpy(sim.m_resource_manager.m_hresults.data(), sim.m_resource_manager.m_dresults, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sim.m_resource_manager.m_htimes.data(), sim.m_resource_manager.m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost); 

    sim.m_path_manager.WriteIntoFiles(sim.m_resource_manager.m_hresults, sim.m_resource_manager.m_htimes, sim.m_path_manager.m_stc_file);
    
    //float x_left = (sim.m_config.GetBeta() - sim.m_config.GetEps()) / (sim.m_config.GetBeta() - sim.m_config.GetAlpha());
    //float x_right = 1;

    //sim.ComputeRateVals(STOCHASTIC, RATE, 0, 100, 0);
    cudaDeviceSynchronize();

    return 0;
}            