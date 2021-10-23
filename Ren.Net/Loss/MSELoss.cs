using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Loss
{
    public class MSELoss
    {
        public Tensor CaculateLoss(Tensor label, Tensor output)
        {
            return output - label;
        }
    }
}
