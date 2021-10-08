using Ren.Net.Objects;
using Ren.Net.Optimizers;
using Serilog;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ren.Net.Networks
{
    public class Linear : NetModule
    {
        public static Dictionary<string, int> RecordDic = new Dictionary<string, int>();
        private readonly Random r = new Random(DateTime.UtcNow.Millisecond);
        /// <summary>
        /// 输入层神经元个数
        /// </summary>
        public int InputNumber { set; get; }
        /// <summary>
        /// 输出层神经元个数
        /// </summary>
        public int OutputNumber { set; get; }
        // public int BatchSize { set; get; }
        /// <summary>
        /// 一层神经元 存储的结构
        /// </summary>
        // public List<NetNeuron> FullyNeurns { set; get; }
        /// <summary>
        /// 权重单独保存在一个地图里面，方向是正向传播的方向，list 每个元素是当前 神经元素的个数，float[] 数组是上一层元素的个数
        /// </summary>
        public List<float[]> WI { set; get; }
        public List<float> WB { set; get; }
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

            WI = new List<float[]>(outputNumber);
            WB = new List<float>(outputNumber);

            int sumInput = outputNumber + inputNumber;

            for (int i = 0; i < outputNumber; i++)
            {
                float[] wiTemp = new float[inputNumber];
                for (int j = 0; j < inputNumber; j++)
                {
                    wiTemp[j] = W_value_method(sumInput);
                }
                WI.Add(wiTemp);
                WB.Add(1);
            }
        }
        public override Torch Forward(Torch @in)
        {
            int batchSize = @in.BatchSize;          // batch 的大小

            if (batchSize == -1)
            {
                throw new Exception("Linear::Forward, batchSize is -1 or neuronNumber is -1");
            }
            Optimizer.InputNumber = this.InputNumber;
            Optimizer.OutputNumber = this.OutputNumber;

            Torch x_out = new Torch(OutputNumber, batchSize);   // 神经元的数量是下一层的大小

            X_In = @in.Clone() as Torch;    // 保存输入

            for (int i = 0; i < OutputNumber; i++)
            {
                for (int j = 0; j < InputNumber; j++)
                {
                    //// ******************test********************
                    //// 下一层的输入 (i) = (i,j)  * 上一层输出 (j)
                    //x_out.Data[i][0] += WI[i][j] * @in.Data[j][0];
                    //continue;
                    //// ******************test********************

                    for (int k = 0; k < batchSize; k++)
                    {
                        x_out.Data[i][k] += WI[i][j] * @in.Data[j][k];
                    }
                }
                // 更新偏置项
                for (int k = 0; k < batchSize; k++)
                {
                    x_out.Data[i][k] += WB[k];
                }
            }
            return x_out;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="out">list 是当前层神经元的数量</param>
        /// <returns></returns>
        public override Torch Backup(Torch @out)    // wi 数量是上一层神经元的数量，假设out 里面 是 误差值
        {
            int batchSize = @out.BatchSize;

            if (batchSize == -1)
            {
                throw new Exception("Linear::Backup, batchSize is -1 or neuronNumber is -1");
            }
            Torch sensitive_out = new Torch(InputNumber, @out.BatchSize);   // list 的个数 表示上一层的神经元个数
            for (int i = 0; i < OutputNumber; i++)
            {
                for (int j = 0; j < InputNumber; j++)
                {
                    //// ******************test********************
                    //// 损失值 上一层神经元 = (i,j) * 下一层神经元 i 
                    //sensitive_out.Data[j][0] += WI[i][j] * @out.Data[i][0];
                    //continue;
                    //// ******************test********************

                    for (int k = 0; k < batchSize; k++)
                    {
                        sensitive_out.Data[j][k] += WI[i][j] * @out.Data[i][k];
                    }
                }
            }
            for (int i = 0; i < OutputNumber; i++)
            {
                for (int j = 0; j < InputNumber; j++)
                {
                    //// ******************test********************
                    //float dwTemp = X_In.Data[j][0] * @out.Data[i][0];
                    //float deltWI = Optimizer.GetOptimizer(dwTemp, i, j);
                    //WI[i][j] -= deltWI;      // list 的个数 表示当前层的神经元个数
                    //// WI[i][j] -= dwTemp * 0.001F;       // list 的个数 表示当前层的神经元个数
                    //continue;
                    //// ******************test********************

                    float[] dwArray = new float[batchSize];
                    for (int k = 0; k < batchSize; k++)
                    {
                        dwArray[k] = X_In.Data[j][k] * @out.Data[i][k];
                    }
                    float dwAverage = dwArray.Average();

                    WI[i][j] -= Optimizer.GetOptimizer(dwAverage, i, j);

                    // ******************test********************
                    lock (RecordDic)
                    {
                        string key = i + "_" + j;
                        if(!RecordDic.ContainsKey(key))
                        {
                            RecordDic[key] = 1;
                        }
                        else
                        {
                            RecordDic[key] += 1;
                        }
                    }
                    // ******************test********************
                }
            }
            PrintWI();

            for (int i = 0; i < @out.Data.Count; i++)
            {
                float dw = @out.Data[i].Average();

                WB[i] -= Optimizer.GetOptimizer(dw, i);
            }

            return sensitive_out;
        }
        /// <summary>
        /// 初始化权值，np.random.randn(n) * sqrt(2.0/n)，遵循 sumInput 个数的正太分布
        /// </summary>
        /// <param name="sumInput">输入个数</param>
        /// <returns></returns>
        public virtual float W_value_method(int sumInput)
        {
            // return 1F;

            //float x = (float)r.NextDouble();
            //float number = (Math.Abs(x) / 1) * (2.0F / sumInput);

            float y = (float)r.NextDouble();
            float x = (float)r.NextDouble();
            float number = (float)(Math.Cos(2 * Math.PI * x) * Math.Sqrt(-2 * Math.Log(1 - y)));
            number *= (float)Math.Sqrt(2.0 / sumInput);
            number = Math.Abs(number);
            number *= (2.0F / sumInput);

            //float y = (float)r.NextDouble();
            //float x = (float)r.NextDouble();
            //float number = (float)(Math.Cos(2 * Math.PI * x) * Math.Sqrt(-2 * Math.Log(1 - y)));
            //number *= (float)Math.Sqrt(2.0 / sumInput);
            //number = Math.Abs(number);

            return number;
        }

        public override void ADDGradient(float epsilon)
        {
            foreach (var item in WI)
            {
                for (int i = 0; i < item.Length; i++)
                {
                    item[i] += epsilon;
                }
            }
        }

        public override void ReduceGradient(float epsilon)
        {
            foreach (var item in WI)
            {
                for (int i = 0; i < item.Length; i++)
                {
                    item[i] -= epsilon;
                }
            }
        }
        private void PrintWI()
        {
            List<string> lines = new List<string>();

            for (int i = 0; i < WI.Count; i++)
            {
                lines.Add(string.Join(" ", WI[i]));
            }
            Log.Debug("WI: " + lines.Count + "\t" + string.Join(" | ", lines));
        }
    }
}
