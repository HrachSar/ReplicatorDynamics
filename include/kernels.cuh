#ifndef KERNELS_CUH
#define KERNELS_CUH


namespace Kernels{
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
};

#endif