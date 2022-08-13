using Ren.Device;
using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.ActivationFunction
{
    [Serializable]
    public class ReLUCUDA : ReLU
    {
        public override DeviceTpye Device { get => DeviceTpye.CUDA; }
        public override void Init()
        {
            X_IN = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            Log.Debug($"ReLU CUDA inited");
        }
        public override Tensor Forward(Tensor @in)
        {
            Tensor.Copy(@in, X_IN);
            Tensor.Relu(X_IN, X_IN, @in);
            return @in;
        }
        public override Tensor Backward(Tensor @out)
        {
            Tensor.Relu(X_IN, @out, @out);
            return @out;
        }
    }
}
