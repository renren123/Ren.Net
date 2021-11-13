using Ren.Net.Objects;
using Ren.Net.Optimizers;
using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Networks
{
    [Serializable]
    public class BatchNorm1D : NetModule
    {
        public static bool IsTrain { set; get; } = true;
        public static readonly float E = 0.00000001F;
        public static readonly float Momentum = 0.9F;
        /// <summary>
        /// 对应Ub
        /// </summary>
        private float RunningMean { set; get; }
        /// <summary>
        /// 对应Sigmab
        /// </summary>
        private float RunningVar { set; get; }
        private float[] x_hat { set; get; }
        private float UbAverage { set; get; }
        private float SigmmaAverage { set; get; }
        public float Ub { set; get; }
        public float Sigmab { set; get; }
        private float Gamma { set; get; } = 1;
        private float Bata { set; get; } = 1;

        public Adam AdamGamma { set; get; }
        public Adam AdamBata { set; get; }
        //public float Vgamma { set; get; }
        //public float Sgamma { set; get; }
        //public float Vbata { set; get; }
        //public float Sbata { set; get; }



        public Tensor X_IN { set; get; }
        public Tensor X_Hat { set; get; }
        /// <summary>
        /// 输入层神经元个数
        /// </summary>
        public int InputNumber { set; get; }
        public float[] UB { set; get; }
        public float[] SIGMAB { set; get; }
        public float[] RMean { set; get; }
        public float[] RVar { set; get; }
        /// <summary>
        /// 第一列元素 Gamma、第二列 Bata
        /// </summary>
        public Tensor GammaBata { set; get; }
        /// <summary>
        /// 第一列元素 UB、第二列 SigmaB
        /// </summary>
        public Tensor UbSigmaB { set; get; }
        /// <summary>
        /// 第一列元素 RMean、第二列 RVar
        /// </summary>
        public Tensor RMeanVar { set; get; }

        private BatchNorm1D BatchNorm1DDevice { set; get; }

        public BatchNorm1D(int inputNumber)
        {
            this.InputNumber = inputNumber;
        }

        public override void Init()
        {
            this.BatchNorm1DDevice = InstenceHelper<BatchNorm1D>.GetInstence(
                typeof(BatchNorm1D),
                new object[] { this.InputNumber }).
                Find(p => p.Device == this.Device);
            //this.Optimizer.InputNumber = InputNumber;
            //this.Optimizer.OutputNumber = 2;

            this.Optimizer.InputNumber = 2;
            this.Optimizer.OutputNumber = InputNumber;

            this.Optimizer.Init();

            this.BatchNorm1DDevice.Optimizer = this.Optimizer;

            this.BatchNorm1DDevice.InputNumber = this.InputNumber;
            this.BatchNorm1DDevice.Init();
        }
        public override Tensor Forward(Tensor @in)
        {
            return this.BatchNorm1DDevice.Forward(@in);
        }
        public override Tensor Backup(Tensor @out)
        {
            return this.BatchNorm1DDevice.Backup(@out);
        }
    }
}
