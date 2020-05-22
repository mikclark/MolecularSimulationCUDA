using System;
using System.Collections.Generic;
using ManagedCuda;

//using System.Linq;

namespace MolecularSimulationUsingCUDA
{
    public class SimulationMolecules
    {

        private readonly int maxCapacity = 0;
        public float[] x;
        public float[] y;
        public float[] z;
        public int[] types;
        public CudaDeviceVariable<float> gpu_x { get; protected set; }
        public CudaDeviceVariable<float> gpu_y { get; protected set; }
        public CudaDeviceVariable<float> gpu_z { get; protected set; }
        public CudaDeviceVariable<int> gpu_types { get; protected set; }

        private SortedSet<int> emptyIndices;

        public SimulationMolecules(int maxCapacity)
        {
            this.maxCapacity = maxCapacity;
            gpu_x = new float[maxCapacity];
            gpu_y = new float[maxCapacity];
            gpu_z = new float[maxCapacity];
            gpu_types = new int[maxCapacity];
            ClearMolecules();
        }
        public SimulationMolecules(SimulationMolecules source) : this(source.x, source.y, source.z, source.types, source.maxCapacity) { }

        public SimulationMolecules(ICollection<float> x, ICollection<float> y, ICollection<float> z, ICollection<int> types, int maxCapacity = -1)
        {
            if (x.Count != y.Count || x.Count != z.Count || x.Count != types.Count)
            {
                throw new Exception("SimulationMolecules constructor failed: the lists \"x\", \"y\", \"z\", and \"types\" must be the same size.");
            }
            if (maxCapacity > -1 && x.Count > maxCapacity)
            {
                throw new Exception($"SimulationMolecules constructor failed: provided maxCapacity {maxCapacity} is smaller than the {x.Count} positions provided.");
            }
            this.maxCapacity = maxCapacity > -1 ? maxCapacity : x.Count;

            this.x = new float[this.maxCapacity];
            this.gpu_x = this.x;
            this.y = new float[this.maxCapacity];
            this.gpu_y = this.y;
            this.z = new float[this.maxCapacity];
            this.gpu_z = this.z;
            this.types = new int[this.maxCapacity];
            this.gpu_types = this.types;
            this.emptyIndices = new SortedSet<int>();

            x.CopyTo(this.x, 0);
            y.CopyTo(this.y, 0);
            z.CopyTo(this.z, 0);
            types.CopyTo(this.types, 0);

            this.CopyAllMoleculesToDevice();

            IEnumerator<int> typesEnumerator = types.GetEnumerator();
            for (int i = 0; i < this.maxCapacity; i++)
            {
                if (!typesEnumerator.MoveNext() || typesEnumerator.Current == -1)
                {
                    emptyIndices.Add(i);
                }
            }
        }

        private void CopyAllMoleculesToDevice()
        {
            this.gpu_x.CopyToDevice(this.x);
            this.gpu_y.CopyToDevice(this.y);
            this.gpu_z.CopyToDevice(this.z);
            this.gpu_types.CopyToDevice(this.types);
        }
        private void CopyOneMoleculeToDevice(int index)
        {
            this.gpu_x[index] = this.x[index];
            this.gpu_y[index] = this.y[index];
            this.gpu_z[index] = this.z[index];
            this.gpu_types[index] = this.types[index];
        }

        private void ClearMolecules()
        {
            if(this.maxCapacity < 0) {
                throw new Exception($"ClearMolecules failed: maxCapacity={this.maxCapacity} is invalid");
            }
            this.x = new float[this.maxCapacity];
            this.y = new float[this.maxCapacity];
            this.z = new float[this.maxCapacity];
            this.types = new int[this.maxCapacity];
            this.emptyIndices = new SortedSet<int>();

            for (int i = 0; i < this.maxCapacity; i++)
            {
                this.x[i] = 0.0F;
                this.y[i] = 0.0F;
                this.z[i] = 0.0F;
                this.types[i] = -1;
                this.emptyIndices.Add(i); 
            }

            this.CopyAllMoleculesToDevice();
        }

        public void AssignRandomPositions(float Lx, float Ly, float Lz, params int[] nByType)
        {
            if (nByType.Length == 0)
            {
                return;
            }

            int nTotal = 0;
            for(int iType = 0; iType < nByType.Length; iType++)
            {
                nTotal += nByType[iType];
            }
            if(nTotal > this.maxCapacity)
            {
                throw new Exception($"AssignRandomPositions failed: requested number {nTotal} of molecules is greater than maxCapacity ({maxCapacity} molecules).");
            }

            // Assign positions
            RandomPCGSharp.Pcg random = new RandomPCGSharp.Pcg();
            this.x = random.NextFloats(nTotal, -0.5F*Lx, 0.5F*Lx);
            this.y = random.NextFloats(nTotal, -0.5F*Ly, 0.5F*Ly);
            this.z = random.NextFloats(nTotal, -0.5F*Lz, 0.5F*Lz);

            // Assign types
            this.types = new int[nTotal]; 
            int currentN = 0;
            for (int iType = 0; iType < nByType.Length; iType++)
            {
                for(int jMolecule = currentN; jMolecule < currentN + nByType[iType]; jMolecule++)
                {
                    this.types[jMolecule] = iType;
                }
                currentN += nByType[iType];
            }
            this.CopyAllMoleculesToDevice();

            // Reset empty indices
            this.emptyIndices = new SortedSet<int>();
            for (int iEmpty = nTotal; iEmpty < this.maxCapacity; iEmpty++)
            {
                this.emptyIndices.Add(iEmpty);
            }
        }
        public void AddMoleculePositions(ICollection<float> x, ICollection<float> y, ICollection<float> z, int type)
        {
            if (x.Count != y.Count || x.Count != z.Count)
            {
                throw new Exception("AddMoleculePositions failed: the lists x, y, and z must be the same size.");
            }
            if(x.Count > emptyIndices.Count)
            {
                throw new Exception($"AddMoleculePositions failed: trying to add {x.Count} molecules, but only {emptyIndices.Count} empty slots remain.");
            }

            IEnumerator<float> newXEnumerator = x.GetEnumerator();
            IEnumerator<float> newYEnumerator = y.GetEnumerator();
            IEnumerator<float> newZEnumerator = z.GetEnumerator();
            while(newXEnumerator.MoveNext() && newYEnumerator.MoveNext() && newZEnumerator.MoveNext())
            {
                AddOneMoleculePosition(newXEnumerator.Current, newYEnumerator.Current, newZEnumerator.Current, type);
            }

        }

        public void AddOneMoleculePosition(float x, float y, float z, int type)
        {
            if(this.emptyIndices.Count == 0)
            {
                throw new Exception("AddOneMoleculePosition failed: no empty slots left.");
            }
            int index = this.emptyIndices.Min;
            this.x[index] = x;
            this.y[index] = y;
            this.z[index] = z;
            this.types[index] = type;
            this.CopyOneMoleculeToDevice(index);
            this.emptyIndices.Remove(index);
        }
        public void RemoveOneMolecule(int index)
        {
            this.x[index] = 0.0F;
            this.y[index] = 0.0F;
            this.z[index] = 0.0F;
            this.types[index] = -1;
            this.CopyOneMoleculeToDevice(index);
            this.emptyIndices.Add(index);
        }

        public void ShiftOneMolecule(int index, float x, float y, float z)
        {
            this.x[index] = x;
            this.y[index] = y;
            this.z[index] = z;
            this.CopyOneMoleculeToDevice(index);
        }

    }
}
