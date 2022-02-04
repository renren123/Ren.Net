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
        public static readonly float E = (float)Math.Pow(10, -5);
        public static readonly float Momentum = 0.9F;

        public Tensor X_IN { set; get; }
        public Tensor X_Hat { set; get; }
        /// <summary>
        /// 输入层神经元个数
        /// </summary>
        public int InputNumber { set; get; }
        public Tensor Gamma { set; get; }
        public Tensor Bata { set; get; }
        /// <summary>
        /// 第一列元素 Gamma、第二列 Bata
        /// </summary>
        // public Tensor GammaBata { set; get; }
        /// <summary>
        /// 第一列元素 UB、第二列 SigmaB
        /// </summary>
        // public Tensor UbSigmaB { set; get; }
        public Tensor UB { set; get; }
        public Tensor SigmaB { set; get; }
        public Tensor RMean { set; get; }
        public Tensor RVar { set; get; }
        /// <summary>
        /// 第一列元素 RMean、第二列 RVar
        /// </summary>
        // public Tensor RMeanVar { set; get; }

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

            this.Optimizer.InputNumber = 1;
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
        public void PrintArray(float[,] array)
        {
            Console.WriteLine();
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write($"{array[i, j]} ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
        public override string ToString()
        {
            return $"BatchNorm1D [{InputNumber}]";
        }
    }
}
