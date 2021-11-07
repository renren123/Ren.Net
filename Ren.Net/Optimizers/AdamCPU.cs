using Ren.Device;
using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    public class AdamCPU : Adam
    {
        public override DeviceTpye Device { get =>  DeviceTpye.CPU; }
        public AdamCPU(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            VTorch = new Tensor(OutputNumber, InputNumber, 0F);
            STorch = new Tensor(OutputNumber, InputNumber, 0F);
        }
        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            VTorch = B1 * VTorch + (1 - B1) * dw;
            STorch = B2 * STorch + (1 - B2) * Tensor.DotMultiplySelf(dw, dw);

            Tensor Vcorrection = VTorch / (1 - B1_Pow);
            Tensor Scorrection = STorch / (1 - B2_Pow);
            Tensor dividend = LearningRate * Vcorrection;
            //Tensor divisor = Tensor.Sqrt(Scorrection) + E;

            Tensor divisor = Scorrection.Sqrt() + E;

            return Tensor.DotDivide(dividend, divisor);
        }
        public override void Step()
        {
            B1_Pow *= B1;
            B2_Pow *= B2;
        }
    }
}
