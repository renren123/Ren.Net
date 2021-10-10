using Ren.Net.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    [Serializable]
    public class NetModule
    {
        public Optimizer Optimizer { set; get; }
        public WIOptimizer WIOptimizer { set; get; }
        public virtual void Init()
        {
            
        }
        public virtual Torch Forward(Torch @in) 
        {
            throw new NotImplementedException();
        }

        public virtual Torch Backup(Torch @out)
        {
            throw new NotImplementedException();
        }
    }
}
