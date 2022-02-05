using Ren.Device;
using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    [Serializable]
    public class AdamCPU : Adam
    {
        public override DeviceTpye Device { get =>  DeviceTpye.CPU; }
        public AdamCPU(float learningRate) : base(learningRate) { }
        /// <summary>
        /// Adam 初始化
        /// </summary>
        public override void Init()
        {
            VTorch = new Tensor(OutputNumber, InputNumber, 0F);
            STorch = new Tensor(OutputNumber, InputNumber, 0F);
            Log.Debug($"Adam CPU inited [{InputNumber}, {OutputNumber}]");
        }
        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            VTorch = B1 * VTorch + (1 - B1) * dw;
            STorch = B2 * STorch + (1 - B2) * Tensor.DotMultiply(dw, dw);

            Tensor Vcorrection = VTorch / (1 - B1_Pow);
            Tensor Scorrection = STorch / (1 - B2_Pow);
            Tensor dividend = LearningRate * Vcorrection;
            Tensor divisor = Scorrection.Sqrt() + E;

            return Tensor.DotDivide(dividend, divisor);
        }
        public override void Step()
        {
            B1_Pow *= B1;
            B2_Pow *= B2;
        }
        public override object Clone()
        {
            AdamCPU adamCPU = new AdamCPU(this.LearningRate)
            {
                B1_Pow = B1_Pow,
                B2_Pow = B2_Pow
            };
            if (VTorch != null)
            {
                adamCPU.VTorch = this.VTorch.Clone() as Tensor;
            }
            if (STorch != null)
            {
                adamCPU.STorch = this.STorch.Clone() as Tensor;
            }
            return adamCPU;
        }
    }
}
