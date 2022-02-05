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
            // ############  test 
            //Minus = output - label;
            //return output - label;


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
                            Tensor.Minus(output, label, Tensor.SwapA);
                            Tensor.DotMultiply(Tensor.SwapA, Tensor.SwapA, output);
                            return output;
                        }
                    default:
                        throw new NotImplementedException();
                }
            }

            //switch (Tensor.Device)
            //{
            //    case DeviceTpye.CPU:
            //        {
            //            return output - label;
            //        }
            //    case DeviceTpye.CUDA:
            //        {
            //            Tensor.Minus(output, label, Tensor.SwapA);
            //            Tensor.Copy(Tensor.SwapA, output);
            //            return output;
            //        }
            //    default:
            //        throw new NotImplementedException();
            //}
        }
        public override Tensor Backup(Tensor output)
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
                            //return Minus;
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
