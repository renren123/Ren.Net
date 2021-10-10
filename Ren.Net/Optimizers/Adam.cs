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

        public Torch VTorch { set; get; }
        public Torch STorch { set; get; }
        public Adam(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            VTorch = new Torch(OutputNumber, InputNumber);
            STorch = new Torch(OutputNumber, InputNumber);
        }
        public override Torch GetOptimizer(Torch dw)
        {
            VTorch = B1 * VTorch + (1 - B1) * dw;
            STorch = B2 * STorch + (1 - B2) * Torch.DotMultiply(dw, dw);

            Torch Vcorrection = VTorch / (1 - B1_Pow);
            Torch Scorrection = STorch / (1 - B2_Pow);
            Torch dividend = LearningRate * Vcorrection;
            Torch divisor = Torch.Sqrt(Scorrection) + E;

            return Torch.DotDivide(dividend, divisor);
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
                adam.VTorch = this.VTorch.Clone() as Torch;
                adam.STorch = this.STorch.Clone() as Torch;
            }
            return adam;
        }
    }
}
