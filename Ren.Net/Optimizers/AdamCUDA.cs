using Ren.Device;
using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    [Serializable]
    public class AdamCUDA : Adam
    {
        private Tensor SwapA { set; get; }
        private Tensor SwapB { set; get; }
        private Tensor SwapC { set; get; }
        public override DeviceTpye Device { get => DeviceTpye.CUDA; }
        public AdamCUDA(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            VTorch = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            STorch = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);

            VTorch.Width = STorch.Width = OutputNumber;
            VTorch.Height = STorch.Height = InputNumber;
            Log.Debug($"Adam CUDA inited [{InputNumber}, {OutputNumber}]");
        }
        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            //Tensor.DotMultiply(dw, dw, Tensor.SwapA);
            //Tensor.Multiply((1 - B2), Tensor.SwapA, Tensor.SwapB);
            //Tensor.Multiply(B2, STorch, Tensor.SwapA);
            //Tensor.Add(Tensor.SwapA, Tensor.SwapB, STorch);     // 计算出 STorch

            //Tensor.Multiply(B1, VTorch, Tensor.SwapA);
            //Tensor.Multiply((1 - B1), dw, Tensor.SwapB);
            //Tensor.Add(Tensor.SwapA, Tensor.SwapB, VTorch);     // 计算出 VTorch

            //Tensor.DotDivide(VTorch, (1 - B1_Pow), Tensor.SwapA);
            //Tensor.Multiply(LearningRate, Tensor.SwapA, Tensor.SwapB);  // 计算出 Tensor.SwapB = dividend
            //Tensor.DotDivide(STorch, (1 - B2_Pow), Tensor.SwapA);
            //Tensor.Sqrt(Tensor.SwapA);
            //Tensor.Add(Tensor.SwapA, E, Tensor.SwapC);          // 计算出 divisor， Tensor.SwapC = divisor
            //Tensor.DotDivide(Tensor.SwapB, Tensor.SwapC, @out);

            //return @out;

            LoadPublicValue();

            Tensor.DotMultiply(dw, dw, SwapA);
            Tensor.Multiply((1 - B2), SwapA, SwapB);
            Tensor.Multiply(B2, STorch, SwapA);
            Tensor.Add(SwapA, SwapB, STorch);     // 计算出 STorch

            Tensor.Multiply(B1, VTorch, SwapA);
            Tensor.Multiply((1 - B1), dw, SwapB);
            Tensor.Add(SwapA, SwapB, VTorch);     // 计算出 VTorch

            Tensor.DotDivide(VTorch, (1 - B1_Pow), SwapA);
            Tensor.Multiply(LearningRate, SwapA, SwapB);  // 计算出 Tensor.SwapB = dividend
            Tensor.DotDivide(STorch, (1 - B2_Pow), SwapA);
            Tensor.Sqrt(SwapA);
            Tensor.Add(SwapA, E, SwapC);          // 计算出 divisor， Tensor.SwapC = divisor
            Tensor.DotDivide(SwapB, SwapC, @out);

            SetPublicValue(SwapA, SwapB, SwapC);
            return @out;
        }
        public override void Step()
        {
            B1_Pow *= B1;
            B2_Pow *= B2;
        }
        public void LoadPublicValue()
        {
            SwapA = NetParameter.GetRegisterParameter();
            SwapB = NetParameter.GetRegisterParameter();
            SwapC = NetParameter.GetRegisterParameter();
            if (SwapA == null || SwapB == null || SwapC == null)
            {
                throw new Exception("can not find enough swap tensor");
            }
        }
    }
}
