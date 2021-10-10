using Ren.Net.Objects;
using Ren.Net.Optimizers;
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
        /// <summary>
        /// 输入层神经元个数
        /// </summary>
        public int InputNumber { set; get; }
        /// <summary>
        /// 输出层神经元个数
        /// </summary>
        public int OutputNumber { set; get; }
        /// <summary>
        /// 权重数组
        /// </summary>
        public Torch WI { set; get; }
        /// <summary>
        /// list 的数量是前一层的数量
        /// </summary>
        public Torch X_In { set; get; }
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
            Optimizer.InputNumber = this.InputNumber + 1;
            Optimizer.OutputNumber = this.OutputNumber;
            Optimizer.Init();

            int sumInput = OutputNumber + InputNumber;
            WI = new Torch(OutputNumber, InputNumber + 1, (int i, int j) =>
            {
                if (j == InputNumber)
                {
                    return 1F;
                }
                else
                {
                    return WIOptimizer.GetWI(sumInput);
                }
            });
            
            Log.Debug($"Linear inited [{InputNumber}, {OutputNumber}]");
        }
        public override Torch Forward(Torch @in)
        {
            int batchSize = @in.Column;          // batch 的大小

            if (batchSize == -1)
            {
                throw new Exception("Linear::Forward, batchSize is -1 or neuronNumber is -1");
            }
            X_In = @in.Clone() as Torch;    // 保存输入，用于反向传播时更新 WI 的大小

            @in = @in.AddOneRowWithValue(batchSize, 1F);

            Torch x_out = WI * @in;

            return x_out;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="out">list 是当前层神经元的数量</param>
        /// <returns></returns>
        public override Torch Backup(Torch @out)    // wi 数量是上一层神经元的数量，假设out 里面 是 误差值
        {
            int batchSize = @out.Column;

            if (batchSize == -1)
            {
                throw new Exception("Linear::Backup, batchSize is -1 or neuronNumber is -1");
            }

            Torch sensitive_out =WI.Transpose() * @out;

            X_In = X_In.AddOneRowWithValue(batchSize, 1F);

            var dwTemp = @out * X_In.Transpose();

            WI -= Optimizer.GetOptimizer(dwTemp);

            sensitive_out = sensitive_out.RemoveLastOneRow();

            return sensitive_out;
        }
        public override string ToString()
        {
            return $"Linear [{InputNumber}, {OutputNumber}]";
        }
    }
}
