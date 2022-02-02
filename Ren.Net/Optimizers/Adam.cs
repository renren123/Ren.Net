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
        }
        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            return AdamDevice.GetOptimizer(dw, @out);
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
            };
            if (this.AdamDevice != null)
            {
                adam.AdamDevice = this.AdamDevice.Clone() as AdamCPU;
            }
            //if(VTorch != null)
            //{
            //    adam.VTorch = this.VTorch.Clone() as Tensor;
            //    adam.STorch = this.STorch.Clone() as Tensor;
            //}
            return adam;
        }
    }
}
