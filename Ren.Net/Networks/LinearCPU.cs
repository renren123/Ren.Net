using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Networks
{
    public class LinearCPU : Linear
    {
        public LinearCPU(int inputNumber, int outputNumber) : base(inputNumber, outputNumber)
        {
            
        }
        public override void Init()
        {
            base.Init();
            int sumInput = OutputNumber + InputNumber;
            WI = new Tensor(OutputNumber, InputNumber + 1, (int i, int j) =>
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

            Log.Debug($"LinearCPU inited [{InputNumber}, {OutputNumber}]");
        }
        public override Tensor Forward(Tensor @in)
        {
            int batchSize = @in.Column;          // batch 的大小
            if (batchSize <= 0)
            {
                throw new Exception($"Linear::Forward, batchSize {batchSize}");
            }

            X_In = @in.AddOneRowWithValue(batchSize, 1F);
            @in = WI * X_In;

            return @in;
        }
        public override Tensor Backup(Tensor @out)
        {
            int batchSize = @out.Column;
            if (batchSize <= 0)
            {
                throw new Exception($"Linear::Backup, batchSize is {batchSize}");
            }
            Tensor sensitive_out = WI.Transpose() * @out;
            Tensor dwTemp = @out * X_In.Transpose();
            WI -= Optimizer.GetOptimizer(dwTemp, null);
            return sensitive_out.RemoveLastOneRow();
        }
    }
}
