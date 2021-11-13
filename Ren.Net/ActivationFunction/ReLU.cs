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
        private ReLU ReLUDevice { set; get; }

        public override void Init()
        {
            ReLUDevice = InstenceHelper<ReLU>.GetInstence(typeof(ReLU), null).Find(p => p.Device == this.Device);
            ReLUDevice.MaxLinearNumber = this.MaxLinearNumber;
            ReLUDevice.Init();
        }
        public ReLU()
        {
            this.WIOptimizer = new ReLUWIOptimizer();
        }
        public override Tensor Forward(Tensor @in)
        {
            return ReLUDevice.Forward(@in);
        }
        public override Tensor Backup(Tensor @out)
        {
            return ReLUDevice.Backup(@out);
        }
        public override string ToString()
        {
            return "ReLU";
        }
    }
}
