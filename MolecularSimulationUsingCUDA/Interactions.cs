using System;
using System.Collections.Generic;
using ManagedCuda;

namespace MolecularSimulationUsingCUDA
{
    public class Interactions
    {
        protected float[] epsilon;
        protected float[] sigma;
        public readonly CudaDeviceVariable<float> cudaEpsilon = new float[100];
        public readonly CudaDeviceVariable<float> cudaSigma = new float[100];

        public Interactions()
        {
            sigma = new float[100];
            epsilon = new float[100];
            cudaSigma = sigma;
            cudaEpsilon = epsilon;
        }
        public Interactions(Interactions i)
        {
            sigma = i.sigma;
            epsilon = i.epsilon;
        }
        public void SetLJParameters(int i, int j, float thisEpsilon, float thisSigma)
        {
            epsilon[i + 10 * j] = thisEpsilon;
            sigma[i + 10 * j] = thisSigma;
            cudaEpsilon[i + 10 * j] = thisEpsilon;
            cudaSigma[i + 10 * j] = thisSigma;
            if (i != j)
            {
                epsilon[j + 10 * i] = thisEpsilon;
                sigma[j + 10 * i] = thisSigma;
                cudaEpsilon[j + 10 * i] = thisEpsilon;
                cudaSigma[j + 10 * i] = thisSigma;
            }
        }
        public float Sigma(int i, int j)
        {
            return this.sigma[i + 10 * j];
        }
        public float Epsilon(int i, int j)
        {
            return this.epsilon[i + 10 * j];
        }

        public void SigmaPointer(out ManagedCuda.BasicTypes.CUdeviceptr sigmaDevicePointer)
        {
            sigmaDevicePointer = cudaSigma.DevicePointer;
        }
        public void EpsilonPointer(out ManagedCuda.BasicTypes.CUdeviceptr epsilonDevicePointer)
        {
            epsilonDevicePointer = cudaEpsilon.DevicePointer;
        }

        public ManagedCuda.BasicTypes.CUdeviceptr SigmaPointer()
        {
            return cudaSigma.DevicePointer;
        }
        public ManagedCuda.BasicTypes.CUdeviceptr EpsilonPointer()
        {
            return cudaEpsilon.DevicePointer;
        }
    }
}
