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
            Tensor.Minus(output, label, Tensor.SwapA);
            Tensor.Copy(Tensor.SwapA, output);
            return output;

            return output - label;
        }
    }
}
