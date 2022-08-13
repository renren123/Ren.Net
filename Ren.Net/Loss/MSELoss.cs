using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;
using Ren.Device;

namespace Ren.Net.Loss
{
    public class MSELoss : NetLoss
    {
        Tensor Minus { set; get; }
        public MSELoss() : base()
        {

        }
        public MSELoss(string reduction) : base(reduction)
        {

        }
        public override Tensor CaculateLoss(Tensor label, Tensor output)
        {
            if (Reduction == "mean")
            {
                throw new NotImplementedException();
            }
            else
            {
                switch (Tensor.Device)
                {
                    case DeviceTpye.CPU:
                        {
                            Minus = output - label;
                            return Tensor.DotMultiply(Minus, Minus);
                        }
                    case DeviceTpye.CUDA:
                        {
                            if (Minus == null)
                            {
                                Minus = new Tensor(Tensor.MaxLinearNumber, Tensor.MaxLinearNumber, 0F);
                            }
                            Tensor.Minus(output, label, Minus);
                            Tensor.DotMultiply(Minus, Minus, output);
                            return output;
                        }
                    default:
                        throw new NotImplementedException();
                }
            }
        }
        public override Tensor Backward(Tensor output)
        {
            int batchSize = output.Column;

            if (Reduction == "mean")
            {
                throw new NotImplementedException();
            }
            else
            {
                switch (Tensor.Device)
                {
                    case DeviceTpye.CPU:
                        {
                            return Minus * (1.0F / (batchSize * 2 ));
                        }
                    case DeviceTpye.CUDA:
                        {
                            Tensor.Multiply((1.0F / batchSize), Minus, output);
                            return output;
                        }
                    default:
                        throw new NotImplementedException();
                }
            }
        }
    }
}
