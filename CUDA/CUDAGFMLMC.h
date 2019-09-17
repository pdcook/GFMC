#ifndef __GFMLMC__
#define __GFMLMC__

#include <thrust/tuple.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>
//#include <stdio.h>
#include <thread>
#include "assert.h"

#define RNGS 200000000 // total number of RNGS before repeat, make big, but small enough to fit on each device
#define _USE_MATH_DEFINES
#define INF 2000000000 // a really big number ~2 billion is the max for int
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }    // inline function used for returning errors that occur in __device__ code since they cannot be seen otherwise

using namespace std;

// global boolean to see if the random number generator has been initialized on each device
extern bool* rand_set;
bool* rand_set;

// global random number generator states for each device
extern curandState** globalDeviceStates;
curandState** globalDeviceStates;

// prints the basic properties of the currently selected gpu
void CUDABasicProperties(int device_id)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    fprintf(stderr, "GPU %d: %s\n", device_id, prop.name);
    double mem = (double)prop.totalGlobalMem/1000000000;
    fprintf(stderr, "MEM: %f GB\n", mem);
    double freq = (double)prop.clockRate/1000000;
    fprintf(stderr, "CLOCK: %f GHZ\n", freq);
    fprintf(stderr, "Compute: %d.%d\n", prop.major, prop.minor);
}

// inline function used for returning errors that occur in __device__ code since they cannot be seen otherwise
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// function for initializing CUDA random number generator for each thread on the device
__global__ void initialize_curand_on_kernels(curandState * state, unsigned long seed)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// function for generating random numbers in CUDA device code
__device__ float RND(curandState* globalState, int ind)
{
    //copy state to local mem
    curandState localState = globalState[ind];
    //apply uniform distribution with calculated random
    float rndval = curand_uniform( &localState );
    //update state
    globalState[ind] = localState;
    //return value
    return rndval;
}

// Snell's law
__device__ thrust::tuple<float,float,float> multilayer_snell(float ni, float nt, float mu_x, float mu_y, float mu_z)
{
    float ti = acosf(abs(mu_z));
    float tt = asinf((ni*sinf(ti))/nt);
    return thrust::make_tuple(mu_x*ni/nt, mu_y*ni/nt, (mu_z/abs(mu_z))*cosf(tt));
}

// function used for determining reflection/transmission
__device__ float fresnel_snell(float ni, float nt, float mu_z)
{
    float ti, tt, R;

    if (abs(mu_z) > 0.99999){ R = pow(((nt-ni)/(nt+ni)),2); }
    else
    {
        ti = acosf(abs(mu_z));
        // if ni*sinf(ti)/nt >=1 then total internal reflection occurs, and thus R = 1
        if ( (ni*sinf(ti))/nt >=1. ) { R= 1.; }
        else
        {
            tt = asinf((ni*sinf(ti))/nt);
            R = 0.5 * ( (pow(sinf(ti-tt),2))/(pow(sinf(ti+tt),2)) + (pow(tan(ti-tt),2))/(pow(tan(ti+tt),2)) );
        }
    }
    return R;
}

// determine if a photon incident on the sample begins to propagate or is reflected
__device__ bool incident_reflection(int idx, curandState* globalState, float mu_z, float* n, float* mu_s, int layer)
{
    // check if the incident layer is glass
    if (mu_s[layer] == 0.0)
    {
        float n1,n2,n3,r1,r2;

        n1 = 1.;
        n2 = n[layer];
        if (mu_z > 0) { n3 = n[layer+1]; }
        else { n3 = n[layer-1]; }

        r1 = fresnel_snell(n1, n2, mu_z);
        r2 = fresnel_snell(n2, n3, mu_z);

        return (RND(globalState, idx) < r1 + (pow((1-r1),2))*r2/(1-r1*r2));

    }

    else
    {
        return (RND(globalState, idx) < fresnel_snell(1., n[layer], mu_z));
    }
}

// calculates the bin ordinal for a value and bin with max value xmax and width dx
__device__ int calc_bin(float u, float umin, float umax, float du) 
{

    // if the value is above the maximum, or below the minimum, place it in the final "extra" bin (-1)
    if (u >= umax || u < umin)
    {   
        return -1; 
    }   

    // if its within the max value, round x/dx towards 0 to int to find the bin ordinal
    else
    {   

        umax = umax-umin;
        u = u - umin;

        return __float2int_rz(u/du);
    }   

}

// single photon propagator
__device__ thrust::tuple<float,float,float,float,float,float> MLMC(int idx, curandState* globalState, float* A, float rmax, float dr, float zmax, float dz, int N, int layers, float* n, float* mu_a, float* mu_s, float* mu_t, float* g, float* t, float* bounds, float w, float x, float y, float z, float mu_x, float mu_y, float mu_z)
{

    // set up initial values
    int N_r = ceil(rmax/dr);
    int N_z = ceil(zmax/dz);

    float Reflected = 0;
    float Transmitted = 0;
    float Extra = 0;
    float threshold = 0.0001;
    int m = 10;
    int layer;
    int nextlayer;
    int rbin, zbin;
    float s, d, deltaW, cos_theta, sin_theta, phi, cos_phi, sin_phi, mu_x_, mu_y_, mu_z_, z_sqrt;

    // define starting layer
    if (mu_z > 0){ layer = 0; }
    else { layer = layers - 1; }

    // propagate while w > 0

    while (w > 0)
    {
        s = -logf(RND(globalState, idx))/mu_t[layer];

        if (mu_z < 0)
        {
            d = (bounds[layer] - z)/mu_z;
            nextlayer = layer - 1;
        }
        else if (mu_z > 0)
        {
            d = (bounds[layer+1] - z)/mu_z;
            nextlayer = layer + 1;
        }
        else if (mu_z == 0)
        {
            d = INF;
            nextlayer = layer;
        }

        // boundary conditions
        while ( s >= d )
        {

            x += d*mu_x;
            y += d*mu_y;
            z += d*mu_z;
            s -= d;

            if (nextlayer == layers)
            {
                if ( RND(globalState, idx) < fresnel_snell(n[layer],1.,mu_z) )
                {
                    // internal reflection
                    mu_z *= -1;
                }
                else                   // photon is transmitted
                {
                    Transmitted += w;
                    // refraction via Snell's Law
                    thrust::tie(mu_x, mu_y, mu_z) = multilayer_snell(n[layer], 1., mu_x, mu_y, mu_z);
                    w = 0;
                    break;
                }
            }

            else if (nextlayer == -1)                 // photon attempts to reflect/backscatter
            {
                if ( RND(globalState, idx) < fresnel_snell(n[layer], 1., mu_z))
                {
                    // photon is internally reflected
                    mu_z *= -1;
                }
                else                   // photon backscatters
                {
                    Reflected += w;
                    // refraction via Snell's Law
                    thrust::tie(mu_x, mu_y, mu_z) = multilayer_snell(n[layer], 1., mu_x, mu_y, mu_z);
                    w = 0;
                    break;
                }
            }
            else
            {
                if (RND(globalState, idx) < fresnel_snell(n[layer], n[nextlayer], mu_z))
                {
                    mu_z *= -1;
                }
                else
                {
                    thrust::tie(mu_x, mu_y, mu_z) = multilayer_snell(n[layer], n[nextlayer], mu_x, mu_y, mu_z);
                    if (mu_s[nextlayer] != 0 && s != 0)
                    {
                        s *= mu_t[layer]/mu_t[nextlayer];
                    }
                    layer = nextlayer;
                }
            }

            if (mu_z < 0)
            {
                d = (bounds[layer] - z)/mu_z;
                nextlayer = layer - 1;
            }
            else if (mu_z > 0)
            {
                d = (bounds[layer+1] - z)/mu_z;
                nextlayer = layer + 1;
            }
            else if (mu_z == 0)
            {
                d = INF;
                nextlayer = layer;
            }

        }

        x += s*mu_x;     //
        y += s*mu_y;     // Hop
        z += s*mu_z;     //

        // partial absorption event
        deltaW = w*mu_a[layer]/mu_t[layer];
        w -= deltaW;
        rbin = calc_bin(sqrt(x*x+y*y),0,rmax,dr);
        zbin = calc_bin(z,0,zmax,dz);
        if ( rbin == -1 || zbin == -1 )
        {
            Extra += deltaW;
        }
        else
        {
            atomicAdd(&(A[zbin*N_r+rbin]), deltaW/N);
        }

        // roullette
        if (w <= threshold)
        {
            if (RND(globalState, idx) <= 1/m){ w*=m; }
            else { w = 0; }
        }

        // scattering event: update the photon's direction cosines only if it's weight isn't 0
        /// Spin ///
        if ( w > 0 )
        {
            if (g[layer] == 0.){ cos_theta = 2*RND(globalState, idx) - 1;}
            else { cos_theta = (1/(2*g[layer]))*(1+g[layer]*g[layer]-pow(((1-g[layer]*g[layer])/(1-g[layer]+2*g[layer]*RND(globalState, idx))),2)); }

            phi = 2 * M_PI * RND(globalState, idx);
            cos_phi = cosf(phi);
            sin_phi = sinf(phi);
            sin_theta = sqrt(1. - pow(cos_theta,2));

            if (abs(mu_z) > 0.99999)
            {
                mu_x_ = sin_theta*cos_phi;
                mu_y_ = sin_theta*sin_phi;
                mu_z_ = (mu_z/abs(mu_z))*cos_theta;
            }
            else
            {
                z_sqrt = sqrt(1 - mu_z*mu_z);
                mu_x_ = sin_theta/z_sqrt*(mu_x*mu_z*cos_phi - mu_y*sin_phi) + mu_x*cos_theta;
                mu_y_ = sin_theta/z_sqrt*(mu_y*mu_z*cos_phi + mu_x*sin_phi) + mu_y*cos_theta;
                mu_z_ = -1.0*sin_theta*cos_phi*z_sqrt + mu_z*cos_theta;
            }

            mu_x = mu_x_;
            mu_y = mu_y_;
            mu_z = mu_z_;
        }
    }

    return thrust::make_tuple(Extra, Reflected, Transmitted, mu_x, mu_y, mu_z);

}

__global__ void CUDAGFMLMC(curandState* globalState, int focustype, float wr, float z0, float l0, float* A, float* R_diffuse, float* R_specular, float* T_diffuse, float* T_direct, float* E, float rmax, float dr, float zmax, float dz, int N, int layers, float* n, float* g, float* t, float* bounds, float* mu_a, float* mu_s, float* mu_t)
{

    int idx = blockIdx.x*blockDim.x+threadIdx.x;        // unique thread identifier used in the CUDA random number generator

    int step = blockDim.x * gridDim.x;                  // step size so each thread knows which photons it is responsible for

    int N_r = ceil(rmax/dr);
    int N_z = ceil(zmax/dz);

    float Extra = 0;
    float Reflected = 0;
    float Transmitted = 0;

    float w, x, y, z, r_i, phi_i, mu_x, mu_y, mu_z;
    float d, deltaW, cos_theta, sin_theta, phi, cos_phi, sin_phi, mu_x_, mu_y_, mu_z_, z_sqrt;
    float threshold = 0.0001;
    int m = 10;
    float wz, R, eta; // stuff for gaussian focusing
    int rbin, zbin;
    int layer = 0;

    // calculate beam waist radius if using gaussian focusing, set it to 0 otherwise
    // l0 must be in nm
    float w0 = 0;
    if (focustype == 1)
    {
        // handle possible floating point error
        if (abs(wr*wr*wr*wr-4*pow(((z0*l0*1e-7)/M_PI/n[0]),2)) < 1e-15)
        {
            w0 = wr / sqrtf(2);
        }
        else
        {
            w0 = sqrt((wr*wr+sqrt(wr*wr*wr*wr-4*pow(((z0*l0*1e-7)/M_PI/n[0]),2)))/2);
        }
    }

    float t_tot = 0;
    for (int i = 0; i < layers; i++)
    {
        t_tot += t[i];
    }

    for (int i = idx; i < N; i += step)
    {
        w = 1;      // initial weight of photon

        // read focus type 0 for traditional 1 for gaussian
        // determine initial depth
        if ( focustype == 0 )
        {
            z = 0;
            wz = wr;
        }
        else if ( focustype == 1 )
        {
            // sample z from 0 to t
            eta = RND(globalState, idx);
            z = -logf(1-eta+eta*exp(-(mu_t[0])*t_tot))/mu_t[0];
            wz = w0*sqrt(1+pow(((l0*1e-7)*(z-z0)/M_PI/n[0]/(w0*w0)),2));
        }

        // determine initial radial position
        phi_i = 2*M_PI*RND(globalState, idx);
        r_i = (wz/sqrtf(2))*sqrt(-logf(RND(globalState, idx)));
        // convert to initial cartesian x and y position
        x = r_i*cosf(phi_i); y = r_i*sinf(phi_i);

        // determine initial direction
        if (focustype == 0)
        {
            mu_x = 0-x; mu_y = 0-y; mu_z = z0-z;
        }
        else if (focustype == 1)
        {
            // determine radius of curvature
            // handle overflow error
            if (abs(z-z0) < 1e-30)
            {
                R = 1e30;
            }
            else
            {
                R = -1*(z-z0)*(1+pow((M_PI*n[0]*w0*w0/((z-z0)*l0*1e-7)),2));
            }

            if ( (z-z0) < 0 )
            {
                mu_x = 0-x, mu_y = 0-y; mu_z = R;
            }
            else
            {
                mu_x = x-0; mu_y = y-0; mu_z = -1*R;
            }
        }
        // normalize initial direction
        d = sqrt(mu_x*mu_x+mu_y*mu_y+mu_z*mu_z);
        mu_x = mu_x/d;
        mu_y = mu_y/d;
        mu_z = mu_z/d;

        // Gaussian propagation must do one extra drop and spin before MC
        if ( focustype == 1 )
        {
            // from the finite z sampling, we have photons that are ballistically transmitted: they never interact with the sample, here is the weight of the current photon that never interacted with the sample and thus was transmitted
            deltaW = w*exp(-mu_t[0]*t_tot);
            if ( abs(mu_z) == 1.)
            {
                atomicAdd(T_direct, deltaW);
            }
            else
            {
                atomicAdd(T_diffuse, deltaW);
            }
            w -= deltaW;

            // partial absorption event
            deltaW = w*mu_a[layer]/mu_t[layer];
            w -= deltaW;
            rbin = calc_bin(sqrt(x*x+y*y),0,rmax,dr);
            zbin = calc_bin(z,0,zmax,dz);
            if (rbin == -1 || zbin == -1)
            {
                atomicAdd(E, deltaW);
            }
            else
            {
                atomicAdd(&(A[zbin*N_r+rbin]), deltaW/N);
            }

            // roullette
            if (w <= threshold)
            {
                if (RND(globalState, idx) <= 1/m){ w*=m; }
                else { w = 0; }
            }

            // scattering event: update the photon's direction cosines only if it's weight isn't 0
            /// Spin ///
            if ( w > 0 )
            {
                if (g[layer] == 0.){ cos_theta = 2*RND(globalState, idx) - 1;}
                else { cos_theta = (1/(2*g[layer]))*(1+g[layer]*g[layer]-pow(((1-g[layer]*g[layer])/(1-g[layer]+2*g[layer]*RND(globalState, idx))),2)); }

                phi = 2 * M_PI * RND(globalState, idx);
                cos_phi = cosf(phi);
                sin_phi = sinf(phi);
                sin_theta = sqrt(1. - pow(cos_theta,2));

                if (abs(mu_z) > 0.99999)
                {
                    mu_x_ = sin_theta*cos_phi;
                    mu_y_ = sin_theta*sin_phi;
                    mu_z_ = (mu_z/abs(mu_z))*cos_theta;
                }
                else
                {
                    z_sqrt = sqrt(1 - mu_z*mu_z);
                    mu_x_ = sin_theta/z_sqrt*(mu_x*mu_z*cos_phi - mu_y*sin_phi) + mu_x*cos_theta;
                    mu_y_ = sin_theta/z_sqrt*(mu_y*mu_z*cos_phi + mu_x*sin_phi) + mu_y*cos_theta;
                    mu_z_ = -1.0*sin_theta*cos_phi*z_sqrt + mu_z*cos_theta;
                }

                mu_x = mu_x_;
                mu_y = mu_y_;
                mu_z = mu_z_;
            }
        }

        if (w > 0)
        {
            // Monte Carlo Photon Transport
            thrust::tie(Extra, Reflected, Transmitted, mu_x, mu_y, mu_z) = MLMC(idx, globalState, A, rmax, dr, zmax, dz, N, layers, n, mu_a, mu_s, mu_t, g, t, bounds, w, x, y, z, mu_x, mu_y, mu_z);

            // catch weight that fell outside of bins
            atomicAdd(E, Extra);

            if ( abs(mu_z) == 1.)
            {
                atomicAdd(R_specular, Reflected);
                atomicAdd(T_direct, Transmitted);
                w = 0;
            }
            else
            {
                atomicAdd(R_diffuse, Reflected);
                atomicAdd(T_diffuse, Transmitted);
                w = 0;
            }
        }

    }

    // convert values to percents and return them
    //return thrust::make_tuple(A/N, R_diffuse/N, R_specular/N, T_diffuse/N, T_direct/N);
}

thrust::tuple<float*,float,float,float,float,float> GFMLMC(int dev, int focustype, float wr, float z0, float l0, float rmax, float dr, float zmax, float dz, int N, int layers, float* n, float* g, float* t, float* bounds, float* mu_a, float* mu_s, float* mu_t)
{
    // gaussian focusing requires single layer sample, for now
    assert(layers == 1);
    // input a value that we know what to do with please
    assert(focustype == 0 || focustype == 1);

    // dictate which GPU to run on
    cudaSetDevice(dev);

    int threadsPerBlock = 512;              // number of threads per block, internet claims that 256 gives best performance...
    int nBlocks = N/threadsPerBlock + 1;    // number of blocks is always the number of photons divided by the number of threads per block then rounded up (+1). so that each GPU thread gets an equal number of photons while still covering every photon

    // only seed the random number generator once per device
    // rand_set must be initialized to false for all devices in the main function
    // of whatever file uses this function
    // Moreover, globalDeviceStates must be initialized in that same main function
    // for all devices
    if (!rand_set[dev])
    {
        //alocate space for each kernels curandState which is used in the CUDA random number generator
        cudaMalloc(&globalDeviceStates[dev], RNGS*sizeof(curandState));

        //call curand_init on each kernel with the same random seed
        //and init the rng states
        initialize_curand_on_kernels<<<nBlocks,threadsPerBlock>>>(globalDeviceStates[dev], unsigned(time(NULL)));

        gpuErrchk( cudaPeekAtLastError() );     // this prints out the last error encountered in the __device__ code (if there was one)
        gpuErrchk( cudaDeviceSynchronize() );   // this waits for the GPU to finish before continuing while also printing out any errors that are encountered in __device__ code

        rand_set[dev] = true;
    }

    curandState* deviceStates = globalDeviceStates[dev];

    // initialize measurement values
    float R_d = 0;
    float R_s = 0;
    float T_d = 0;
    float T_u = 0;
    float E   = 0;

    // initialize absorptance array
    int N_r = ceil(rmax/dr);
    int N_z = ceil(zmax/dz);
    float* A = new float[N_r*N_z]();    // 1D array which will be mapped to a 2D array with iz*N_r+ir
    // copy measurement values to GPU
    float* dev_A;
    cudaMalloc(&dev_A, N_r*N_z*sizeof(float));
    cudaMemcpy(dev_A, A, N_r*N_z*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_R_d;
    cudaMalloc(&dev_R_d, sizeof(float));
    cudaMemcpy(dev_R_d, &R_d, sizeof(float), cudaMemcpyHostToDevice);
    float* dev_R_s;
    cudaMalloc(&dev_R_s, sizeof(float));
    cudaMemcpy(dev_R_s, &R_s, sizeof(float), cudaMemcpyHostToDevice);
    float* dev_T_d;
    cudaMalloc(&dev_T_d, sizeof(float));
    cudaMemcpy(dev_T_d, &T_d, sizeof(float), cudaMemcpyHostToDevice);
    float* dev_T_u;
    cudaMalloc(&dev_T_u, sizeof(float));
    cudaMemcpy(dev_T_u, &T_u, sizeof(float), cudaMemcpyHostToDevice);
    float* dev_E;
    cudaMalloc(&dev_E, sizeof(float));
    cudaMemcpy(dev_E, &E, sizeof(float), cudaMemcpyHostToDevice);

    // copy sample properties to GPU
    float* dev_n;
    cudaMalloc(&dev_n, layers*sizeof(float));
    cudaMemcpy(dev_n, n, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_g;
    cudaMalloc(&dev_g, layers*sizeof(float));
    cudaMemcpy(dev_g, g, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_t;
    cudaMalloc(&dev_t, layers*sizeof(float));
    cudaMemcpy(dev_t, t, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_bounds;
    cudaMalloc(&dev_bounds, (layers+1)*sizeof(float));
    cudaMemcpy(dev_bounds, bounds, (layers+1)*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_mu_a;
    cudaMalloc(&dev_mu_a, layers*sizeof(float));
    cudaMemcpy(dev_mu_a, mu_a, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_mu_s;
    cudaMalloc(&dev_mu_s, layers*sizeof(float));
    cudaMemcpy(dev_mu_s, mu_s, layers*sizeof(float), cudaMemcpyHostToDevice);
    float* dev_mu_t;
    cudaMalloc(&dev_mu_t, layers*sizeof(float));
    cudaMemcpy(dev_mu_t, mu_t, layers*sizeof(float), cudaMemcpyHostToDevice);

    // run the simulation on the GPU
    CUDAGFMLMC<<<nBlocks,threadsPerBlock>>>(deviceStates, focustype, wr, z0, l0, dev_A, dev_R_d, dev_R_s, dev_T_d, dev_T_u, dev_E, rmax, dr, zmax, dz, N, layers, dev_n, dev_g, dev_t, dev_bounds, dev_mu_a, dev_mu_s, dev_mu_t);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // copy the results back
    cudaMemcpy(A, dev_A, N_r*N_z*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&R_d, dev_R_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&R_s, dev_R_s, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&T_d, dev_T_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&T_u, dev_T_u, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&E, dev_E, sizeof(float), cudaMemcpyDeviceToHost);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // free GPU memory
    cudaFree(dev_A); cudaFree(dev_E); cudaFree(dev_R_d); cudaFree(dev_R_s); cudaFree(dev_T_d); cudaFree(dev_T_u); cudaFree(dev_n); cudaFree(dev_g); cudaFree(dev_t); cudaFree(dev_bounds); cudaFree(dev_mu_a); cudaFree(dev_mu_s); cudaFree(dev_mu_t);


    return thrust::make_tuple(A, E/N, R_d/N, R_s/N, T_d/N, T_u/N);
}

#endif
