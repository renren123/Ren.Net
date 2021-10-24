using Ren.Net.Objects;
using Ren.Net.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.ActivationFunction
{
    [Serializable]
    public class ReLU : NetModule
    {
        private Tensor X_IN;
        public ReLU()
        {
            this.WIOptimizer = new ReLUWIOptimizer();
        }
        public override Tensor Forward(Tensor @in)
        {
            //X_IN = @in.Clone() as Tensor;

            //for (int i = 0; i < @in.Row; i++)
            //{
            //    for (int j = 0; j < @in.Column; j++)
            //    {
            //        if(@in[i, j] < 0)
            //        {
            //            @in[i, j] = 0;
            //        }
            //    }
            //}
            //return @in;

            X_IN = @in.Clone() as Tensor;

            return @in.Relu(X_IN);
        }
        public override Tensor Backup(Tensor @out)
        {
            //for (int i = 0; i < X_IN.Row; i++)
            //{
            //    for (int j = 0; j < X_IN.Column; j++)
            //    {
            //        if (X_IN[i, j] < 0)
            //        {
            //            @out[i, j] = 0;
            //        }
            //    }
            //}
            //return @out;

            return @out.Relu(X_IN);
        }
        public override string ToString()
        {
            return "ReLU";
        }
    }
}
