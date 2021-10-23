using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    [Serializable]
    public class Optimizer : ICloneable
    {
        public float LearningRate { set; get; }
        public int InputNumber { set; get; } = -1;
        public int OutputNumber { set; get; } = -1;

        public Optimizer(float learningRate)
        {
            this.LearningRate = learningRate;
        }
        public virtual void Init()
        {
            throw new NotImplementedException();
        }
        //public virtual float GetOptimizer(float dw,int OutputIndex, int InputIndex)
        //{
        //    throw new NotImplementedException();
        //}

        public virtual Tensor GetOptimizer(Tensor dw)
        {
            throw new NotImplementedException();
        }


        public virtual void Step()
        {
            throw new NotImplementedException();
        }
        public virtual object Clone()
        {
            throw new NotImplementedException();
        }
    }
}
