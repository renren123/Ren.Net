using Ren.Device;
using Ren.Net.Objects;
using Ren.Net.Optimizers;
using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.ActivationFunction
{
    [Serializable]
    public class ReLU : NetModule
    {
        public Tensor X_IN;
        public virtual DeviceTpye Device { get; set; }
        private ReLU ReLUDevice { set; get; }

        public override void Init()
        {
            ReLUDevice = InstenceHelper<ReLU>.GetInstence(typeof(ReLU), null).Find(p => p.Device == this.Device);
            ReLUDevice.MaxLinearNumber = this.MaxLinearNumber;
            ReLUDevice.Init();
            return;
            X_IN = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
        }
        public ReLU()
        {
            this.WIOptimizer = new ReLUWIOptimizer();
        }
        public override Tensor Forward(Tensor @in)
        {
            return ReLUDevice.Forward(@in);
            switch (@in.Device)
            {
                case DeviceTpye.CPU:
                    {
                        X_IN = @in.Clone() as Tensor;

                        return @in.Relu(X_IN);
                    }
                    break;
                case DeviceTpye.CUDA:
                    {
                        Tensor.Copy(@in, X_IN);
                        Tensor.Relu(X_IN, X_IN, @in);
                        return @in;
                    }
                    break;
                default:
                    throw new Exception("ReLU::Forward ");
            }
            Tensor.Copy(@in, X_IN);
            Tensor.Relu(X_IN, X_IN, @in);
            return @in;


            if (X_IN != null)
            {
                X_IN.Dispose();
            }
            X_IN = @in.Clone() as Tensor;

            Tensor x_out = @in.Relu(X_IN);
            @in.Dispose();

            return x_out;
        }
        public override Tensor Backup(Tensor @out)
        {
            return ReLUDevice.Backup(@out);
            switch (@out.Device)
            {
                case DeviceTpye.CPU:
                    {
                        return @out.Relu(X_IN);
                    }
                    break;
                case DeviceTpye.CUDA:
                    {
                        Tensor.Relu(X_IN, @out, @out);
                        return @out;
                    }
                    break;
                default:
                    throw new Exception("ReLU::Forward ");
            }

            Tensor.Relu(X_IN, @out, @out);
            return @out;

            Tensor x_out = @out.Relu(X_IN);
            @out.Dispose();

            return x_out;
        }
        public override string ToString()
        {
            return "ReLU";
        }
    }
}
