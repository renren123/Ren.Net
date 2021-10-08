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
            throw new NotImplementedException();
        }

        public virtual Torch Backup(Torch @out)
        {
            throw new NotImplementedException();
        }

        public virtual void ADDGradient(float epsilon)
        {
            throw new NotImplementedException();
        }

        public virtual void ReduceGradient(float epsilon)
        {
            throw new NotImplementedException();
        }
    }
}
