#include "../include/simulator.cuh"

float A[2] = {0};

__global__ void Kernels::SimulateDynamics(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times){
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
__global__ void Kernels::SimulateDynamicsPeriodic(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times){
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
__global__ void Kernels::SolveDynamicsDeterministic(unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times){
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

SimulatorConfig::SimulatorConfig(float alpha, float beta, float rate, 
                        float eps, float dt, float tmin, float tmax){
    this->m_alpha = alpha;
    this->m_beta = beta;
    this->m_rate = rate;
    this->m_eps = eps;
    this->m_dt = dt;
    this->m_tmin = tmin;
    this->m_tmax = tmax;
}

__host__ float SimulatorConfig::GetAlpha(){
    return m_alpha;
}
__host__ float SimulatorConfig::GetBeta(){
    return m_beta;
}
__host__ float SimulatorConfig::GetEps(){
    return m_eps;
}
__host__ float SimulatorConfig::GetRate(){
    return m_rate;
}
__host__ float SimulatorConfig::GetDt(){
    return m_dt;
}
__host__ float SimulatorConfig::GetTmax(){
    return m_tmax;
}
__host__ float SimulatorConfig::GetTmin(){
    return m_tmin;
}
__host__ void SimulatorConfig::SetAlpha(float alpha){
    m_alpha = alpha;
}
__host__ void SimulatorConfig::SetBeta(float beta){
    m_beta = beta;
}
__host__ void SimulatorConfig::SetRate(float rate){
    m_rate = rate;
}
__host__ void SimulatorConfig::SetEps(float eps){
    m_eps = eps;
}
__host__ void SimulatorConfig::SetDt(float dt){
    m_dt = dt;
}
__host__ void SimulatorConfig::SetTmin(float tmin){
    m_tmin = tmin;
}
__host__ void SimulatorConfig::SetTmax(float tmax){
    m_tmax = tmax;
}
SimulatorConfig::~SimulatorConfig(){}

PathManager::PathManager(std::string_view results_deterministic, std::string_view results_periodic, std::string_view results_stochastic){
    this->m_results_deterministic = results_deterministic;
    this->m_results_periodic = results_periodic;
    this->m_results_stochastic = results_stochastic;
    this->m_det_file.open(m_results_deterministic.data(), std::ios::out | std::ios::trunc);
    this->m_stc_file.open(m_results_stochastic.data(), std::ios::out | std::ios::trunc);
    this->m_per_file.open(m_results_periodic.data(), std::ios::out | std::ios::trunc);

    if(!m_det_file || !m_per_file || !m_stc_file)
            throw std::runtime_error("Failed to open output files.");
}

PathManager::PathManager(PathManager&& other) noexcept
    : m_det_file(std::move(other.m_det_file))
    , m_stc_file(std::move(other.m_stc_file))
    , m_per_file(std::move(other.m_per_file))
    , m_results_deterministic(other.m_results_deterministic)
    , m_results_periodic(other.m_results_periodic)
    , m_results_stochastic(other.m_results_stochastic)
{}

__host__ std::string_view PathManager::GetDetPath(){
    return m_results_deterministic;
}
__host__ std::string_view PathManager::GetStcPath(){
    return m_results_stochastic;
}
__host__ std::string_view PathManager::GetPerPath(){
    return m_results_periodic;
}
__host__ void PathManager::SetDetPath(std::string_view path){
    m_results_deterministic = path;
}
__host__ void PathManager::SetPerPath(std::string_view path){
    m_results_periodic = path;
}
__host__ void PathManager::SetStcPath(std::string_view path){
    m_results_stochastic = path;
}
__host__ void PathManager::WriteIntoFiles(std::vector<float>& res, std::vector<float>& times, std::fstream& stream){
    if(!stream.is_open())
        throw std::exception("File stream is not open.");

    for(int i = 0; i < res.size(); i++){
        stream << res[i] << "\t" << times[i] << "\n";
    }

    stream.flush();
}
PathManager::~PathManager(){
    m_det_file.close();
    m_per_file.close();
    m_stc_file.close();
}


__host__ void ResourceManager::AllocateMemory(){
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

    m_hresults.resize(N, 0);
    m_htimes.resize(N, 0);
}
__host__ void ResourceManager::FreeMemory(){
    cudaFree(m_dresults);
    cudaFree(m_dtimes);
    m_hresults.clear();
    m_htimes.clear();
}
ResourceManager::ResourceManager(){
    AllocateMemory();
}
ResourceManager::ResourceManager(ResourceManager&& other) noexcept
    : m_hresults(std::move(other.m_hresults))
    , m_htimes(std::move(other.m_htimes))
    , m_dresults(other.m_dresults)
    , m_dtimes(other.m_dtimes)
{
    other.m_dresults = nullptr;
    other.m_dtimes   = nullptr;
}

ResourceManager::~ResourceManager(){
    FreeMemory();
}

Simulator::Simulator(int blocks, int threads, SimulatorConfig& config, 
                     PathManager& path_manager, ResourceManager& resource_manager)
    : m_blocks(blocks)
    , m_threads(threads)
    , m_config(config)                   
    , m_path_manager(std::move(path_manager))
    , m_resource_manager(std::move(resource_manager))
{
    A[0] = config.GetAlpha();
    A[1] = config.GetBeta();
    cudaMemcpyToSymbol(d_A, A, 2 * sizeof(float)); //for constant memory
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

__host__ float Simulator::ComputeMean(std::vector<float>& res){
    float total = 0.0f;
    for(auto iter = res.begin(); iter != res.end(); iter++){
        total += *iter;
    }
    return total / res.size();
}

__host__ void Simulator::ComputeRateValsPeriodic(int start, int end){
    int j = start;
    std::vector<float> res = std::vector<float>(end - start, 0.0f);
    std::vector<float> time = std::vector<float>(end - start, 0.0f);

    for(;j < end; j++){
        cudaError_t err = cudaMemset(m_resource_manager.m_dtimes, 0, N * sizeof(float));
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 
        float i = (j + 1) * 0.1f;
        Kernels::SimulateDynamicsPeriodic<<<m_blocks, m_threads>>>(12345ULL, m_config.GetTmin(), m_config.GetTmax(), m_config.GetEps(), i, m_config.GetDt(), m_resource_manager.m_dresults, m_resource_manager.m_dtimes);
        cudaDeviceSynchronize();

        err = cudaMemcpy(m_resource_manager.m_htimes.data(), m_resource_manager.m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 

        res[j - start] = i;
        time[j - start] = ComputeMean(m_resource_manager.m_htimes);
    }
    this->m_path_manager.WriteIntoFiles(res, time, m_path_manager.m_per_file);
}
__host__ void Simulator::ComputeRateValsDeterministic(int start, int end){

    int j = start;
    std::vector<float> res = std::vector<float>(end - start, 0.0f);
    std::vector<float> time = std::vector<float>(end - start, 0.0f);

    for(;j < end; j++){
        cudaError_t err = cudaMemset(m_resource_manager.m_dtimes, 0, N * sizeof(float));
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 
        float i = (j + 1) * 0.1f;
        Kernels::SolveDynamicsDeterministic<<<m_blocks, m_threads>>>(12345ULL, m_config.GetTmin(), m_config.GetTmax(), m_config.GetEps(), i, m_config.GetDt(), m_resource_manager.m_dresults, m_resource_manager.m_dtimes);
        cudaDeviceSynchronize();

        err = cudaMemcpy(m_resource_manager.m_htimes.data(), m_resource_manager.m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 

        res[j - start] = i;
        time[j - start] = ComputeMean(m_resource_manager.m_htimes);
    }
    this->m_path_manager.WriteIntoFiles(res, time, m_path_manager.m_det_file);
}
__host__ void Simulator::ComputeRateValsStochastic(int start, int end){
    int j = start;
    std::vector<float> res = std::vector<float>(end - start, 0.0f);
    std::vector<float> time = std::vector<float>(end - start, 0.0f);

    for(;j < end; j++){
        cudaError_t err = cudaMemset(m_resource_manager.m_dtimes, 0, N * sizeof(float));
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 
        float i = (j + 1) * 0.1f;
        Kernels::SimulateDynamics<<<m_blocks, m_threads>>>(12345ULL, m_config.GetTmin(), m_config.GetTmax(), m_config.GetEps(), i, m_config.GetDt(), m_resource_manager.m_dresults, m_resource_manager.m_dtimes);
        cudaDeviceSynchronize();
        err = cudaMemcpy(m_resource_manager.m_htimes.data(), m_resource_manager.m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 

        res[j - start] = i;
        time[j - start] = ComputeMean(m_resource_manager.m_htimes);
    }
    this->m_path_manager.WriteIntoFiles(res, time, m_path_manager.m_stc_file);
}
__host__ void Simulator::SetBlockSize(int size){
    m_threads = size;
}
__host__ void Simulator::SetGridSize(int size){
    m_blocks = size;
}
__host__ int Simulator::GetNumBlocks(){
    return m_blocks;
}
__host__ int Simulator::GetNumThreads(){
    return m_threads;
}
Simulator::~Simulator(){}