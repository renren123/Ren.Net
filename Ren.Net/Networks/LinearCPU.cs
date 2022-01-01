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
        public override Tensor Forward(Tensor @in)
        {
            int batchSize = @in.Column;          // batch 的大小
            X_In = @in.AddOneRowWithValue(batchSize, 1F);
            @in = WI * X_In;

            return @in;
            //return @in.RemoveLastOneRow();
        }
        public override Tensor Backup(Tensor @out)
        {
            //int batchSize = @out.Column;
            //@out = @out.AddOneRowWithValue(batchSize, 1F);
            //X_In = X_In.AddOneRowWithValue(batchSize, 1F);

            Tensor sensitive_out = WI.Transpose() * @out;
            Tensor dwTemp = @out * X_In.Transpose();
            WI -= Optimizer.GetOptimizer(dwTemp, null);
            return sensitive_out.RemoveLastOneRow();
        }
    }
}
