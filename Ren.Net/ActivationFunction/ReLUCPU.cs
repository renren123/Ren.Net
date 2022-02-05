using Ren.Device;
using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.ActivationFunction
{
    [Serializable]
    public class ReLUCPU : ReLU
    {
        public override DeviceTpye Device { get => DeviceTpye.CPU;}
        public override void Init()
        {
            Log.Debug($"ReLU CPU inited");
        }
        public override Tensor Forward(Tensor @in)
        {
            X_IN = @in.Clone() as Tensor;

            for (int i = 0; i < @in.Row; i++)
            {
                for (int j = 0; j < @in.Column; j++)
                {
                    if (@in[i, j] < 0)
                    {
                        @in[i, j] = 0;
                    }
                }
            }
            return @in;

            Tensor result = @in.Relu(X_IN);
            return result;
        }
        public override Tensor Backup(Tensor @out)
        {
            for (int i = 0; i < @out.Row; i++)
            {
                for (int j = 0; j < @out.Column; j++)
                {
                    if (X_IN[i, j] < 0)
                    {
                        @out[i, j] = 0;
                    }
                }
            }
            return @out;

            Tensor result = @out.Relu(X_IN);
            return result;
        }
    }
}
