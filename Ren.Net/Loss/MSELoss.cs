using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Loss
{
    public class MSELoss
    {
        public Torch CaculateLoss(Torch label, Torch output)
        {
            return output - label;
        }
    }
}
