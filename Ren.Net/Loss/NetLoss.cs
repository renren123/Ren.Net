using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Loss
{
    public class NetLoss : NetModule
    {
        public string Reduction { set; get; }
        public NetLoss()
        {
        }
        public NetLoss(string reduction = "")
        {
            this.Reduction = reduction;
        }
        public virtual Tensor CaculateLoss(Tensor label, Tensor output) 
        {
            throw new NotImplementedException();
        }
    }
}
