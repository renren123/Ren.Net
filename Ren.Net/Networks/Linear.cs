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
        public Torch WI { set; get; }
        /// <summary>
        /// list 的数量是前一层的数量
        /// </summary>
        public Torch X_In { set; get; }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputSize"></param>
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
            X_In = @in.Clone() as Torch;    // 保存输入
            //Optimizer.InputNumber = this.InputNumber + 1;
            //Optimizer.OutputNumber = this.OutputNumber;

            //OptimizerTemp = Optimizer.Clone() as Adam;

            //#region old
            //Torch x_out = new Torch(OutputNumber, batchSize);   // 神经元的数量是下一层的大小

            //for (int i = 0; i < OutputNumber; i++)
            //{
            //    for (int j = 0; j < InputNumber; j++)
            //    {
            //        for (int k = 0; k < batchSize; k++)
            //        {
            //            x_out[i, k] += WI[i, j] * @in[j, k];
            //        }
            //    }
            //}
            //for (int i = 0; i < OutputNumber; i++)
            //{
            //    for (int k = 0; k < batchSize; k++)
            //    {
            //        x_out[i, k] += WI[i, InputNumber];
            //    }
            //}
            //#endregion

            #region new
            @in = @in.AddOneRowWithValue(batchSize, 1F);
            Torch x_out = WI * @in;
            #endregion

            //if(!x_out.Equals(x_out_temp))
            //{
            //    Log.Error("Forward is not equal\rn" + x_out + "\r\n\r\n ********************* \r\n\r\n" + x_out_temp);
            //}
            return x_out;
        }
        /// <summary>
        /// 
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
            //Torch sensitive_out = new Torch(InputNumber, batchSize);   // list 的个数 表示上一层的神经元个数

            //for (int i = 0; i < OutputNumber; i++)
            //{
            //    for (int j = 0; j < InputNumber; j++)
            //    {
            //        for (int k = 0; k < batchSize; k++)
            //        {
            //            sensitive_out[j, k] += WI[i, j] * @out[i, k];
            //        }
            //    }
            //}

            Torch sensitive_out =WI.Transpose() * @out;

            //float[,] dwold = new float[OutputNumber, InputNumber];

            //var WI_temp = WI.Clone() as Torch;

            //for (int i = 0; i < OutputNumber; i++)
            //{
            //    for (int j = 0; j < InputNumber; j++)
            //    {
            //        float[] dwArray = new float[batchSize];
            //        for (int k = 0; k < batchSize; k++)
            //        {
            //            dwArray[k] = X_In[j, k] * @out[i, k];
            //        }
            //        float dwAverage = dwArray.Average();

            //        WI[i, j] -= Optimizer.GetOptimizer(dwAverage, i, j);

            //        dwold[i, j] = dwAverage;
            //    }
            //}

            X_In = X_In.AddOneRowWithValue(batchSize, 1F);
            var dwTemp = @out * X_In.Transpose();

            //Torch dwOptimizer = new Torch(OutputNumber, InputNumber + 1, (int i, int j) =>
            //    Optimizer.GetOptimizer(dwTemp[i, j], i, j));
            //WI -= dwOptimizer;


            Parallel.For(0, OutputNumber, (xp) =>
            {
                int i = xp;
                for (int j = 0; j < InputNumber + 1; j++)
                {
                    WI[i, j] -= Optimizer.GetOptimizer(dwTemp[i, j], i, j);
                }
            });

            //for (int i = 0; i < OutputNumber; i++)
            //{
            //    for (int j = 0; j < InputNumber + 1; j++)
            //    {
            //        WI[i, j] -= Optimizer.GetOptimizer(dwTemp[i, j], i, j);
            //    }
            //}

            // Log.Information("WI_temp\r\n" + WI_temp);

            // 更新 WB
            //for (int i = 0; i < @out.Row; i++)
            //{
            //    float dw = @out.RowAverage(i);
            //    WI[i, InputNumber] -= Optimizer.GetOptimizer(dw, i);
            //}

            //for (int i = 0; i < @out.Row; i++)
            //{
            //    float dw = @out.RowAverage(i);

            //    WB[i] -= Optimizer.GetOptimizer(dw, i);
            //}
            // Log.Information("WI\r\n" + WI);

            sensitive_out = sensitive_out.RemoveLastOneRow();

            return sensitive_out;
        }
        public override string ToString()
        {
            return $"Linear [{InputNumber}, {OutputNumber}]";
        }
    }
}
