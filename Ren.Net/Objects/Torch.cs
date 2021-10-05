using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    public class Torch
    {
        public float[] Data1d { set; get; }
        public float[,] Data2d { set; get; }
        public float[,,] Data3d { set; get; }
        public int Dimension =>
            Data1d == null ? 1 :
            Data2d == null ? 2 :
            Data3d == null ? 3 : 0;

        public Torch(float[] data)
        {
            this.Data1d = data;
        }
        public Torch(float[,] data)
        {
            this.Data2d = data;
        }
        public Torch(float[,,] data)
        {
            this.Data3d = data;
        }
    }
}
