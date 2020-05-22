using ManagedCuda;
using ManagedCuda.BasicTypes;
using System.Collections.Generic;
using System;
using System.Linq;

namespace MolecularSimulationUsingCUDA
{
    class Program
    {
        // Settings specific to my GPU device
        const int THREADS_PER_BLOCK = 512;
        const int BLOCKS_PER_GRID = 1024;


        static readonly string PTX_NAME = AppendToExePath("lj.ptx");
        static CudaKernel energyOfExistingMolecule;

        static string AppendToExePath(string fileName)
        {
            string exePath = System.Reflection.Assembly.GetEntryAssembly().Location;
            string exeDir = System.IO.Path.GetDirectoryName(exePath);
            string filePath = System.IO.Path.Combine(exeDir, fileName);
            return filePath;
        }


        static void InitKernels()
        {
            CudaContext context = new CudaContext();
            energyOfExistingMolecule = BuildKernelFromFunction("EnergyOfExistingMolecule", ref context);
        }
        static CudaKernel BuildKernelFromFunction(string functionName, ref CudaContext context)
        {
            //CudaContext newContext = new CudaContext();
            CudaKernel kernel = context.LoadKernelPTX(PTX_NAME, functionName);
            kernel.BlockDimensions = THREADS_PER_BLOCK;
            kernel.GridDimensions = BLOCKS_PER_GRID;
            return kernel;
        }





        

        static void Main(string[] args)
        {
            InitKernels();
            int n = 0x1000000;
            float[] xx = new float[n];
            float[] yy = new float[n];
            float[] zz = new float[n];
            int[] tt = new int[n];

            float L = 512;
            float Lx = L, Ly = L, Lz = L;
            CudaDeviceVariable<float> gpu_lengths = new float[] { Lx, Ly, Lz };
            for (int i = 0; i < n; i++)
            {
                xx[i] = ((float)(i & 0x0000FF) / (float)0x000100 - 0.5F) * Lx;
                yy[i] = ((float)(i & 0x00FF00) / (float)0x010000 - 0.5F) * Ly;
                zz[i] = ((float)(i & 0xFF0000) / (float)0x1000000 - 0.5F) * Lz;
                tt[i] = 1;
            }
            SimulationMolecules simulationMolecules = new SimulationMolecules(xx,yy,zz,tt);

            for (int i = 0; i < 10; i++)
                Console.WriteLine($"{i}  =  ({simulationMolecules.x[i]},{simulationMolecules.y[i]},{simulationMolecules.z[i]})");
            int ii = 0x808080;
            Console.WriteLine($"{ii}  =  ({simulationMolecules.x[ii]},{simulationMolecules.y[ii]},{simulationMolecules.z[ii]})");

            Interactions interactions = new Interactions();
            interactions.SetLJParameters(1, 1, 1.5F, 1.5F);

            float[] cachedEnergies = new float[n / THREADS_PER_BLOCK + 1];
            CudaDeviceVariable<float> gpu_energies = cachedEnergies;

            energyOfExistingMolecule.Run(n,
                simulationMolecules.gpu_x.DevicePointer,
                simulationMolecules.gpu_y.DevicePointer,
                simulationMolecules.gpu_z.DevicePointer,
                simulationMolecules.gpu_types.DevicePointer,
                interactions.SigmaPointer(),
                interactions.EpsilonPointer(),
                gpu_lengths.DevicePointer,
                2.5F,
                0x808080,
                gpu_energies.DevicePointer
                );

            gpu_energies.CopyToHost(cachedEnergies);
            double e = cachedEnergies.Sum();
            Console.WriteLine($"Energy = {e}");
            Console.Read();
        }
    }
}