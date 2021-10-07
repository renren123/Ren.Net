using Ren.Net.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    public class NetModule
    {
        public Optimizer Optimizer { set; get; }
        public virtual Torch Forward(Torch @in) 
        {
            return null;
        }

        public virtual Torch Backup(Torch @out)
        {
            return null;
        }

        public virtual void ADDGradient(float epsilon)
        {
        }

        public virtual void ReduceGradient(float epsilon)
        {
        }
    }
}
