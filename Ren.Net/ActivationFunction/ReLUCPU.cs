using Ren.Device;
using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.ActivationFunction
{
    public class ReLUCPU : ReLU
    {
        public override DeviceTpye Device { get => DeviceTpye.CPU;}
        public override void Init()
        {
            //base.Init();
        }
        public override Tensor Forward(Tensor @in)
        {
            X_IN = @in.Clone() as Tensor;
            Tensor result = @in.Relu(X_IN);
            return result;
        }
        public override Tensor Backup(Tensor @out)
        {
            Tensor result = @out.Relu(X_IN);
            return result;
        }
    }
}
