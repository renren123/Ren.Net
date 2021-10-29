using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    /// <summary>
    /// 原理：https://www.jianshu.com/p/aebcaf8af76e
    /// adam 优化器
    /// </summary>
    [Serializable]
    public class Adam : Optimizer
    {
        public static readonly float E = 0.00000001F;
        public static readonly float B1 = 0.9F;
        public static readonly float B2 = 0.999F;

        private float B1_Pow { set; get; } = B1;
        private float B2_Pow { set; get; } = B2;

        public Tensor VTorch { set; get; }
        public Tensor STorch { set; get; }
        public Adam(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            VTorch = new Tensor(OutputNumber, InputNumber);
            STorch = new Tensor(OutputNumber, InputNumber);
        }
        public override Tensor GetOptimizer(Tensor dw)
        {
            using Tensor dotSquare = Tensor.DotMultiply(dw, dw);
            using Tensor b1_VTorch = B1 * VTorch;
            using Tensor r_B1_VTorch = (1 - B1) * dw;
            using Tensor b2_STorch = B2 * STorch;
            using Tensor r_B2_STorch = (1 - B2) * dotSquare;

            if (VTorch != null)
            {
                VTorch.Dispose();
            }
            VTorch = b1_VTorch + r_B1_VTorch;
            if (STorch != null)
            {
                STorch.Dispose();
            }
            STorch = b2_STorch + r_B2_STorch;
            
            using Tensor Vcorrection = VTorch / (1 - B1_Pow);
            using Tensor Scorrection = STorch / (1 - B2_Pow);
            using Tensor dividend = LearningRate * Vcorrection;
            using Tensor storchSqrt = Tensor.Sqrt(Scorrection);
            using Tensor divisor = storchSqrt + E;

            return Tensor.DotDivide(dividend, divisor);



            //VTorch = B1 * VTorch + (1 - B1) * dw;
            //STorch = B2 * STorch + (1 - B2) * Tensor.DotMultiply(dw, dw);

            //Tensor Vcorrection = VTorch / (1 - B1_Pow);
            //Tensor Scorrection = STorch / (1 - B2_Pow);
            //Tensor dividend = LearningRate * Vcorrection;
            //Tensor divisor = Tensor.Sqrt(Scorrection) + E;

            //return Tensor.DotDivide(dividend, divisor);
        }

        public override void Step()
        {
            B1_Pow *= B1;
            B2_Pow *= B2;
        }
        public override object Clone()
        {
            Adam adam = new Adam(this.LearningRate)
            {
                OutputNumber = this.OutputNumber,
                InputNumber = this.InputNumber,
            };
            if(VTorch != null)
            {
                adam.VTorch = this.VTorch.Clone() as Tensor;
                adam.STorch = this.STorch.Clone() as Tensor;
            }
            return adam;
        }
    }
}
