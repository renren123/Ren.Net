using Ren.Device;
using Ren.Net.Objects;
using Ren.Net.Optimizers;
using Ren.Net.Util;
using Serilog;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Ren.Net.Networks
{
    [Serializable]
    public class Linear : NetModule
    {
        //public static Tensor SwapA { set; get; }
        //public static Tensor SwapB { set; get; }

        /// <summary>
        /// 输入层神经元个数
        /// </summary>
        public int InputNumber { set; get; }
        /// <summary>
        /// 输出层神经元个数
        /// </summary>
        public int OutputNumber { set; get; }
        /// <summary>
        /// 权重数组, 
        /// 数组横坐标的个数是下一列神经元的个数，
        /// 数组纵坐标的个数是上一列神经元的额个数再加一列wb
        /// </summary>
        public Tensor WI { set; get; }
        /// <summary>
        /// list 的数量是前一层的数量
        /// </summary>
        public Tensor X_In { set; get; }

        public Linear LinearDevice { set; get; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputSize">输入层 神经元数量</param>
        /// <param name="outputSize">当前层 神经元的数量</param>
        public Linear(int inputNumber, int outputNumber)
        {
            this.InputNumber = inputNumber;
            this.OutputNumber = outputNumber;
        }
        public override void Init()
        {
            this.LinearDevice = InstenceHelper<Linear>.GetInstence(
                typeof(Linear), 
                new object[] { this.InputNumber, this.OutputNumber }).
                Find(p => p.Device == this.Device);

            Optimizer.LoadNetParameter(NetParameter);
            Optimizer.InputNumber = this.InputNumber + 1;
            Optimizer.OutputNumber = this.OutputNumber;
            Optimizer.Init();

            this.LinearDevice.LoadNetParameter(NetParameter);
            this.LinearDevice.Optimizer = Optimizer;
            this.LinearDevice.WIInitialize = WIInitialize;
            this.LinearDevice.MaxLinearNumber = this.MaxLinearNumber;
            this.LinearDevice.Init();

            //if(SwapA == null && Device == DeviceTpye.CUDA)
            //{
            //    SwapA = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            //}
            //if(SwapB == null && Device == DeviceTpye.CUDA)
            //{
            //    SwapB = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            //}
        }
        public override Tensor Forward(Tensor @in)
        {
            int batchSize = @in.Height == 0 ? @in.Column : @in.Height;          // batch 的大小

            if (batchSize <= 0 || InputNumber != @in.Width)
            {
                throw new Exception($"Linear::Forward, batchSize is {batchSize}, or InputNumber: {InputNumber} != @in.Width: {@in.Width}");
            }
            return this.LinearDevice.Forward(@in);
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="out">list 是当前层神经元的数量</param>
        /// <returns></returns>
        public override Tensor Backup(Tensor @out)    // wi 数量是上一层神经元的数量，假设out 里面 是 误差值
        {
            // batchSize
            if (@out.Column <= 0)
            {
                throw new Exception($"Linear::Backup, batchSize is {@out.Column}");
            }
            return this.LinearDevice.Backup(@out);
        }
        public override string ToString()
        {
            return $"Linear [{InputNumber}, {OutputNumber}]";
        }
    }
}
