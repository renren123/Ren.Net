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
            if (X_IN != null)
            {
                X_IN.Dispose();
            }
            X_IN = @in.Clone() as Tensor;

            Tensor x_out = @in.Relu(X_IN);
            @in.Dispose();

            return x_out;
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

            Tensor x_out = @out.Relu(X_IN);
            @out.Dispose();

            return x_out;
        }
        public override string ToString()
        {
            return "ReLU";
        }
    }
}
