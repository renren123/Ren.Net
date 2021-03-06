using Ren.Device;
using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Networks
{
    [Serializable]
    public class LinearCPU : Linear
    {
        public override DeviceTpye Device { get => DeviceTpye.CPU; }

        public LinearCPU(int inputNumber, int outputNumber) : base(inputNumber, outputNumber)
        {
            
        }
        public override void Init()
        {
            int sumInput = OutputNumber + InputNumber;
            WI = new Tensor(OutputNumber, InputNumber + 1, (int i, int j) =>
            {
                // bias 放到最后一列
                if (j == InputNumber)
                {
                    return 1F;
                }
                else
                {
                    return WIInitialize.GetWI(sumInput);
                }
            });

            Log.Debug($"Linear CPU inited [{InputNumber}, {OutputNumber}]");
        }
        /// <summary>
        /// 正向传播参考
        /// https://www.zybuluo.com/hanbingtao/note/476663
        /// https://cloud.tencent.com/developer/article/1056429
        /// </summary>
        /// <param name="in"></param>
        /// <returns></returns>
        public override Tensor Forward(Tensor @in)
        {
            int batchSize = @in.Column;          // batch 的大小
            X_In = @in.AddLastOneRowWithValue(1F);
            @in = WI * X_In;

            return @in;
        }
        public override Tensor Backup(Tensor @out)
        {
            int batchSize = @out.Column;

            Tensor sensitive_out = WI.Transpose() * @out;
            Tensor dwTemp = @out * X_In.Transpose() * batchSize;

            WI -= Optimizer.GetOptimizer(dwTemp, null); 
            return sensitive_out.RemoveLastOneRow();
        }
    }
}
