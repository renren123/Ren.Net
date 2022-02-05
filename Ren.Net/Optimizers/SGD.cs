using Ren.Net.Objects;
using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    public class SGD : Optimizer
    {
        private SGD SGDDevice { set; get; }
        public SGD(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            SGDDevice = InstenceHelper<SGD>.GetInstence(typeof(SGD), new object[] { LearningRate }).Find(p => p.Device == Device);
            SGDDevice.InputNumber = this.InputNumber;
            SGDDevice.OutputNumber = this.OutputNumber;
            SGDDevice.LearningRate = this.LearningRate;
            SGDDevice.MaxLinearNumber = this.MaxLinearNumber;
            SGDDevice.Init();
        }
        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            return SGDDevice.GetOptimizer(dw, @out);
        }
        public override void Step()
        {
        }
        public override object Clone()
        {
            SGD sgd = new SGD(LearningRate)
            {
                Device = this.Device
            };
            if (SGDDevice != null) 
            {
                sgd.SGDDevice = SGDDevice.Clone() as SGDCPU;
            }
            return sgd;
        }
    }
}
