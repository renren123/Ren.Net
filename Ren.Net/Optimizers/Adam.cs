using Ren.Device;
using Ren.Net.Objects;
using Ren.Net.Util;
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

        public float B1_Pow { set; get; } = B1;
        public float B2_Pow { set; get; } = B2;
        public Tensor VTorch { set; get; }
        public Tensor STorch { set; get; }

        private Adam AdamDevice { set; get; }

        public Adam(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            AdamDevice = InstenceHelper<Adam>.GetInstence(typeof(Adam), new object[] { LearningRate }).Find(p => p.Device == Device);
            AdamDevice.InputNumber = this.InputNumber;
            AdamDevice.OutputNumber = this.OutputNumber;
            AdamDevice.LearningRate = this.LearningRate;
            AdamDevice.MaxLinearNumber = this.MaxLinearNumber;
            AdamDevice.Init();
            return;


            VTorch = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            STorch = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);

            VTorch.Width = STorch.Width = OutputNumber;
            VTorch.Height = STorch.Height = InputNumber;
        }
        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            return AdamDevice.GetOptimizer(dw, @out);


            switch (Tensor.Device)
            {
                case DeviceTpye.CPU:
                    {
                        VTorch = B1 * VTorch + (1 - B1) * dw;
                        STorch = B2 * STorch + (1 - B2) * Tensor.DotMultiplySelf(dw, dw);

                        Tensor Vcorrection = VTorch / (1 - B1_Pow);
                        Tensor Scorrection = STorch / (1 - B2_Pow);
                        Tensor dividend = LearningRate * Vcorrection;
                        //Tensor divisor = Tensor.Sqrt(Scorrection) + E;

                        Tensor divisor = Scorrection.Sqrt() + E;

                        return Tensor.DotDivide(dividend, divisor);



                        //using Tensor dotSquare = Tensor.DotMultiplySelf(dw, dw);
                        //using Tensor b1_VTorch = B1 * VTorch;
                        //using Tensor r_B1_VTorch = (1 - B1) * dw;
                        //using Tensor b2_STorch = B2 * STorch;
                        //using Tensor r_B2_STorch = (1 - B2) * dotSquare;
                        //VTorch = b1_VTorch + r_B1_VTorch;
                        //STorch = b2_STorch + r_B2_STorch;

                        //using Tensor Vcorrection = VTorch / (1 - B1_Pow);
                        //using Tensor Scorrection = STorch / (1 - B2_Pow);
                        //using Tensor dividend = LearningRate * Vcorrection;

                        //Tensor.Sqrt(Scorrection);

                        ////using Tensor storchSqrt = Tensor.Sqrt(Scorrection);
                        //using Tensor divisor = Scorrection + E;
                        //return Tensor.DotDivide(dividend, divisor);
                    }
                    break;
                case DeviceTpye.CUDA:
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
                    }
                    break;
            }
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

            return @out;

            //using Tensor dotSquare = Tensor.DotMultiplySelf(dw, dw);
            //using Tensor b1_VTorch = B1 * VTorch;
            //using Tensor r_B1_VTorch = (1 - B1) * dw;
            //using Tensor b2_STorch = B2 * STorch;
            //using Tensor r_B2_STorch = (1 - B2) * dotSquare;

            //if (VTorch != null)
            //{
            //    VTorch.Dispose();
            //}
            //VTorch = b1_VTorch + r_B1_VTorch;
            //if (STorch != null)
            //{
            //    STorch.Dispose();
            //}
            //STorch = b2_STorch + r_B2_STorch;
            
            //using Tensor Vcorrection = VTorch / (1 - B1_Pow);
            //using Tensor Scorrection = STorch / (1 - B2_Pow);
            //using Tensor dividend = LearningRate * Vcorrection;
            //using Tensor storchSqrt = Tensor.Sqrt(Scorrection);
            //using Tensor divisor = storchSqrt + E;
            //return Tensor.DotDivide(dividend, divisor);


            // ########################## old ##########################
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
            //B1_Pow *= B1;
            //B2_Pow *= B2;
            AdamDevice?.Step();
        }
        public override object Clone()
        {
            Adam adam = new Adam(this.LearningRate)
            {
                OutputNumber = this.OutputNumber,
                InputNumber = this.InputNumber,
                Device = this.Device,
                AdamDevice = this.AdamDevice
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
