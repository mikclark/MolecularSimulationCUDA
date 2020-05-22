#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>


extern "C" {
    __global__ void EnergyOfExistingMolecule(
        const int NTotal,
        const float* x,
        const float* y,
        const float* z,
        const int* types,
        const float* sigma10x10,
        const float* epsilon10x10,
        const float* lengths,
        const float cutoffFactor,
        const int nthMolecule,
        float* cacheEnergy
        )
    {
        __shared__ float localCache[1024];
        unsigned int threadId = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned int cacheIndex = threadIdx.x;
        const float iX = x[nthMolecule];
        const float iY = y[nthMolecule];
        const float iZ = z[nthMolecule];
        const int iType = types[nthMolecule];
        const float Lx = lengths[0];
        const float Ly = lengths[1];
        const float Lz = lengths[2];

        float dx = 0.0, dy = 0.0, dz = 0.0, dr2 = 0.0, idr6 = 0.0, ljEnergy = 0.0;
        int jType = -1;
        float tempEnergy = 0;
        const float cutoffFactor2 = cutoffFactor * cutoffFactor;
        while (threadId < NTotal)
        {
            // Do not calculate the energy of the nthMolecule with itself
            if (threadId == nthMolecule) {
                threadId += blockDim.x * gridDim.x;
                continue;
            }

            jType = types[threadId];
            // Skip "empty" molecules, defined by a negative " types" value
            if (jType < 0) {
                threadId += blockDim.x * gridDim.x;
                continue;
            }
            const float isigma2 = sigma10x10[iType + 10 * jType] * sigma10x10[iType + 10 * jType];
            const float iepsilon = epsilon10x10[iType + 10 * jType];

            dx = x[threadId] - iX;
            dy = y[threadId] - iY;
            dz = z[threadId] - iZ;
            dx = dx - Lx * round(dx / Lx);
            dx = dx - Ly * round(dy / Ly);
            dx = dx - Lz * round(dz / Lz);
            dr2 = (dx * dx + dy * dy + dz * dz);
            dr2 /= isigma2;
            if (dr2 <= cutoffFactor2) {
                idr6 = 1.0 / (dr2 * dr2 * dr2);
                ljEnergy = 4.0 * iepsilon * idr6 * (idr6 - 1.0);
                tempEnergy += ljEnergy;
            }
            threadId += blockDim.x * gridDim.x;
        }
        localCache[cacheIndex] = tempEnergy;
        __syncthreads();

        int i = blockDim.x / 2;
        while (i != 0)
        {
            if (cacheIndex < i)
                localCache[cacheIndex] += localCache[cacheIndex + i];
            __syncthreads();
            i /= 2;
        }
        if (cacheIndex == 0)
            cacheEnergy[blockIdx.x] = localCache[0];
    }

    __global__ void VectorSum(const int N, const float* v, float* sum)
    {
        __shared__ float chache[1024];
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned int chacheindex = threadIdx.x;

        float temp = 0;
        while (tid < N)
        {
            temp += v[tid];
            tid += blockDim.x * gridDim.x;
        }
        chache[chacheindex] = temp;
        __syncthreads();

        int i = blockDim.x / 2;
        while (i != 0)
        {
            if (chacheindex < i)
                chache[chacheindex] += chache[chacheindex + i];
            __syncthreads();
            i /= 2;
        }
        if (chacheindex == 0)
            sum[blockIdx.x] = chache[0];
    }

    __global__ void VectorDotProduct (const int N, const float* V1, const float* V2, float* V3)
    {
        __shared__ float chache[1024];
        float temp;
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned int chacheindex = threadIdx.x;

        while (tid < N)
        {
            temp += V1[tid] * V2[tid];
            tid += blockDim.x * gridDim.x;
        }
        chache[chacheindex] = temp;
        __syncthreads();

        int i = blockDim.x / 2;
        while (i != 0)
        {
            if (chacheindex < i)
                chache[chacheindex] += chache[chacheindex + i];
            __syncthreads();
            i /= 2;
        }
        if (chacheindex == 0)
            V3[blockIdx.x] = chache[0];
    }
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %zu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 1; i <= 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i - 1]);
    for (int i = 1; i <= 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i - 1]);
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %zu\n", devProp.totalConstMem);
    printf("Texture alignment:             %zu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

int main()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }

    printf("\nPress any key to exit...");
    char c;
    scanf("%c", &c);

    return 0;
}