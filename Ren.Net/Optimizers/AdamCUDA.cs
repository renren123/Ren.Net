using Ren.Device;
using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    public class AdamCUDA : Adam
    {
        public override DeviceTpye Device { get => DeviceTpye.CUDA; }
        public AdamCUDA(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            VTorch = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            STorch = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);

            VTorch.Width = STorch.Width = OutputNumber;
            VTorch.Height = STorch.Height = InputNumber;
        }
        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            Tensor.DotMultiply(dw, dw, Tensor.SwapA);
            Tensor.Multiply((1 - B2), Tensor.SwapA, Tensor.SwapB);
            Tensor.Multiply(B2, STorch, Tensor.SwapA);
            Tensor.Add(Tensor.SwapA, Tensor.SwapB, STorch);     // 计算出 STorch

            Tensor.Multiply(B1, VTorch, Tensor.SwapA);
            Tensor.Multiply((1 - B1), dw, Tensor.SwapB);
            Tensor.Add(Tensor.SwapA, Tensor.SwapB, VTorch);     // 计算出 VTorch

            Tensor.DotDivide(VTorch, (1 - B1_Pow), Tensor.SwapA);
            Tensor.Multiply(LearningRate, Tensor.SwapA, Tensor.SwapB);  // 计算出 Tensor.SwapB = dividend
            Tensor.DotDivide(STorch, (1 - B2_Pow), Tensor.SwapA);
            Tensor.Sqrt(Tensor.SwapA);
            Tensor.Add(Tensor.SwapA, E, Tensor.SwapC);          // 计算出 divisor， Tensor.SwapC = divisor
            Tensor.DotDivide(Tensor.SwapB, Tensor.SwapC, @out);

            return @out;
        }
        public override void Step()
        {
            B1_Pow *= B1;
            B2_Pow *= B2;
        }
    }
}
