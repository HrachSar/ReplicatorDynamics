#include "../include/simulator.cuh"

float A[2] = {0};

__global__ void Kernels::SimulateDynamics(State state, unsigned long long seed, float tmin, float tmax, float eps, float rate, float dt, float *results, float *times, float *hit, float init_val, bool compute_times){
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= N)
        return;

    curandState rng;
    curand_init(seed, id, 0, &rng);  
    float x;
    if(init_val > 0 && init_val < 1)
        x = init_val;
    else
        x = curand_uniform(&rng); 
    float t = tmin; 

    switch(state){
        case STOCHASTIC:
	        while(t < tmax){

    	    	float u  = curand_uniform(&rng);
	        	float dg_step = 0;
                float t0 = -logf(u + 1e-9f);
	        	float etta_temp = (curand_uniform(&rng) > 0.5f) ? eps : -eps;

    	    	while(dg_step < t0){
	        		float df_step = Simulator::MidpointSolverf(x, etta_temp, dt);
	        		dg_step += 2 * rate * dt;
	        		x += df_step;
                    if(x > 0.99 && compute_times){
                        hit[id]++;
                        times[id] = t + dg_step / (2 * rate);
                        t = tmax;
                        break;
                    }
	        	}
	        	t += t0 / (2 * rate);

    	    }
            break;
        case PERIODIC:
            float u = curand_uniform(&rng);
            float half_period = 1.0f / rate;
            float eps_temp = eps;     
            const float t_final = tmax + u * (2.0f / rate);
    
            while(t < t_final){
                eps_temp = -eps_temp;
            
                float t_next = t + half_period;
                if (t_next > t_final)
                    t_next = t_final;
            
                while(t < t_next){
                    float df_step = Simulator::MidpointSolverf(x, eps_temp, dt);
                    t += dt;
                    x += df_step;
                    if(x > 0.99){
                        hit[id]++;
                        times[id] = t;
                        t = tmax;
                        break;
                    }
                }
            }
            break;
        case DETERMINISTIC:
            while(t < tmax){

                float df_step = Simulator::MidpointSolverf(x, 0, dt);
                x += df_step;

                if(x > 0.99){
                    hit[id]++;
                    times[id] = t;
                    t = tmax;
                    break;
                }
                t += dt;
            }  
            break;
        default:
            break;
    }
    results[id] = x;
}

SimulatorConfig::SimulatorConfig(float alpha, float beta, float rate, 
                        float eps, float dt, float tmin){
    this->m_alpha = alpha;
    this->m_beta = beta;
    this->m_rate = rate;
    this->m_eps = eps;
    this->m_dt = dt;
    this->m_tmin = tmin;
}

__host__ float SimulatorConfig::GetAlpha() const{
    return m_alpha;
}
__host__ float SimulatorConfig::GetBeta() const{
    return m_beta;
}
__host__ float SimulatorConfig::GetEps() const{
    return m_eps;
}
__host__ float SimulatorConfig::GetRate() const{
    return m_rate;
}
__host__ float SimulatorConfig::GetDt() const{
    return m_dt;
}
__host__ float SimulatorConfig::GetTmax() const{
    return m_tmax;
}
__host__ float SimulatorConfig::GetTmin() const{
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

__host__ std::string_view PathManager::GetDetPath() const{
    return m_results_deterministic;
}
__host__ std::string_view PathManager::GetStcPath() const{
    return m_results_stochastic;
}
__host__ std::string_view PathManager::GetPerPath() const{
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
__host__ void PathManager::WriteIntoFiles(std::vector<float>& res, std::vector<float>& times, std::vector<float>& hits, std::fstream& stream){
    if(!stream.is_open())
        throw std::exception("File stream is not open.");

    for(int i = 0; i < res.size(); i++){
        stream << res[i] << "\t" << times[i] << "\t" << hits[i] << "\n";
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

    err = cudaMalloc((void **)&(this->m_dhits), N * sizeof(float));
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    m_hresults.resize(N, 0);
    m_htimes.resize(N, 0);
    m_hhits.resize(N, 0);
}
__host__ void ResourceManager::FreeMemory(){
    cudaFree(m_dresults);
    cudaFree(m_dtimes);
    cudaFree(m_dhits);
    m_hresults.clear();
    m_htimes.clear();
    m_hhits.clear();
}
ResourceManager::ResourceManager(){
    AllocateMemory();
}
ResourceManager::ResourceManager(ResourceManager&& other) noexcept
    : m_hresults(std::move(other.m_hresults))
    , m_htimes(std::move(other.m_htimes))
    , m_hhits(std::move(other.m_hhits))
    , m_dresults(other.m_dresults)
    , m_dtimes(other.m_dtimes)
    , m_dhits(other.m_dhits)
{
    other.m_dresults = nullptr;
    other.m_dtimes   = nullptr;
    other.m_dhits    = nullptr;
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
    float tmax = 100.0f * MyMax(std::abs(1.0f / (config.GetAlpha() + config.GetEps())), std::abs(1.0f / (config.GetAlpha() - config.GetEps())), 
                    std::abs(1.0f / (config.GetBeta() + config.GetEps())), std::abs(1.0f / (config.GetBeta() - config.GetEps())), 
                            std::abs(config.GetAlpha() / ((config.GetAlpha() - config.GetEps())*(config.GetBeta() - config.GetEps()))), std::abs(config.GetAlpha() / ((config.GetAlpha() + config.GetEps())*(config.GetAlpha() + config.GetEps()))));
    m_config.SetTmax(tmax);
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

__host__ void Simulator::ComputeRateVals(State state, Sim_Type type, int start, int end, float x){
    int j = start;
    std::vector<float> res = std::vector<float>(end - start, 0.0f);
    std::vector<float> time = std::vector<float>(end - start, 0.0f);
    std::vector<float> hitProb = std::vector<float>(end - start, 0.0f);

    for(;j < end; j++){
        cudaError_t err = cudaMemset(m_resource_manager.m_dtimes, 0, N * sizeof(float));
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 
        err = cudaMemset(m_resource_manager.m_dhits, 0, N * sizeof(float));
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 
        if(type == RATE){
            float i = (j + 1) * 0.1f;
            Kernels::SimulateDynamics<<<m_blocks, m_threads>>>(state, 12345ULL, m_config.GetTmin(), m_config.GetTmax(), m_config.GetEps(), i, m_config.GetDt(), m_resource_manager.m_dresults, m_resource_manager.m_dtimes, m_resource_manager.m_dhits, x, true);
            res[j - start] = i;
        }else if(type == POSITION){
            float x_star = (m_config.GetBeta() + m_config.GetEps()) / (m_config.GetBeta() - m_config.GetAlpha()) + 0.001;
            x =  x_star + ((m_config.GetBeta() - m_config.GetEps()) / (m_config.GetBeta() - m_config.GetAlpha()) - x_star) * (0.01 * j);
            Kernels::SimulateDynamics<<<m_blocks, m_threads>>>(state, 12345ULL, m_config.GetTmin(), m_config.GetTmax(), m_config.GetEps(), m_config.GetRate(), m_config.GetDt(), m_resource_manager.m_dresults, m_resource_manager.m_dtimes, m_resource_manager.m_dhits, x, true);
            res[j - start] = x;
        }
        cudaDeviceSynchronize();

        err = cudaMemcpy(m_resource_manager.m_htimes.data(), m_resource_manager.m_dtimes, N * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 

        err = cudaMemcpy(m_resource_manager.m_hhits.data(), m_resource_manager.m_dhits, N * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        } 

        time[j - start] = ComputeMean(m_resource_manager.m_htimes);
        hitProb[j - start] = ComputeMean(m_resource_manager.m_hhits);
    }
    switch (state)
    {
    case PERIODIC:
        this->m_path_manager.WriteIntoFiles(res, time, hitProb, m_path_manager.m_per_file);
        break;
    case STOCHASTIC:
        this->m_path_manager.WriteIntoFiles(res, time, hitProb, m_path_manager.m_stc_file);
        break;
    case DETERMINISTIC:
        this->m_path_manager.WriteIntoFiles(res, time, hitProb, m_path_manager.m_det_file);
        break;
    default:
        break;
    }    
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