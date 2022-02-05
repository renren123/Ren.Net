using Ren.Device;
using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    public class SGDCPU : SGD
    {
        public override DeviceTpye Device { get => DeviceTpye.CPU; }
        public SGDCPU(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            Log.Debug($"SGD CPU inited learningRate: {LearningRate}");
        }
        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            return LearningRate * dw;
        }
        public override object Clone()
        {
            return new SGDCPU(this.LearningRate);
        }
    }
}
