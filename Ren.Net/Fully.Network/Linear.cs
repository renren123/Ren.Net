using Ren.Net.Objects;
using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Fully.Network
{
    public class Linear : NetModule
    {
        private Random r;
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

        public List<float[]> V { set; get; }
        public List<float[]> S { set; get; }

        public Torch X_In { set; get; }
        public float StudyRate { set; get; } = 0.0001F;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputSize"></param>
        /// <param name="outputSize">当前层 神经元的数量</param>
        public Linear(int inputNumber, int outputNumber)
        {
            r = new Random(DateTime.UtcNow.Millisecond);
            this.InputNumber = inputNumber;
            this.OutputNumber = outputNumber;

            //FullyNeurns = new List<NetNeuron>(OutputNumber);
            //for (int i = 0; i < OutputNumber; i++)
            //{
            //    FullyNeurns.Add(new NetNeuron()
            //    {
            //        // Wi = new float[OutputNumber],
            //        // SensitiveValue = new float[outputSize]  // 这个是不是要重新初始化
            //    });
            //}
            // ******************************* 初始化 wi *******************************
            WI = new List<float[]>(outputNumber);
            V = new List<float[]>(outputNumber);
            S = new List<float[]>(outputNumber);

            for (int i = 0; i < outputNumber; i++)
            {
                float[] wiTemp = new float[inputNumber];
                for (int j = 0; j < inputNumber; j++)
                {
                    wiTemp[j] = W_value_method(outputNumber);
                }
                WI.Add(wiTemp);

                V.Add(new float[inputNumber]);
                S.Add(new float[inputNumber]);
            }
        }
        public override Torch Forward(Torch @in)
        {
            int batchSize = @in.BatchSize;          // batch 的大小

            if (batchSize == -1)
            {
                throw new Exception("Linear::Forward, batchSize is -1 or neuronNumber is -1");
            }
            Torch x_out = new Torch(OutputNumber, batchSize);

            X_In = @in.Clone() as Torch;    // 保存输入

            for (int i = 0; i < OutputNumber; i++)
            {
                for (int j = 0; j < InputNumber; j++)
                {
                    for (int k = 0; k < batchSize; k++)
                    {
                        x_out.Data[i][k] += WI[i][j] * @in.Data[j][k];
                    }
                }
            }
            return x_out;
        }
        public override Torch Backup(Torch @out)    // wi 数量是上一层神经元的数量，假设out 里面 是 误差值
        {
            int batchSize = @out.BatchSize;

            if (batchSize == -1)
            {
                throw new Exception("Linear::Backup, batchSize is -1 or neuronNumber is -1");
            }

            Torch x_out = new Torch(InputNumber, @out.BatchSize);
            for (int i = 0; i < OutputNumber; i++)
            {
                for (int j = 0; j < InputNumber; j++)
                {
                    for (int k = 0; k < batchSize; k++)
                    {
                        x_out.Data[j][k] += WI[i][j] * @out.Data[i][k];
                    }
                }
            }

            for (int i = 0; i < OutputNumber; i++)
            {
                for (int j = 0; j < InputNumber; j++)
                {
                    float[] dwArray = new float[batchSize];
                    for (int k = 0; k < batchSize; k++)
                    {
                        dwArray[k] = X_In.Data[j][k] * x_out.Data[j][k];
                    }
                    float dwAverage = dwArray.Average();

                    V[i][j] = (float)(AgentClass.B1 * V[i][j] + (1 - AgentClass.B1) * dwAverage);
                    S[i][j] = (float)(AgentClass.B1 * S[i][j] + (1 - AgentClass.B1) * dwAverage * dwAverage);

                    float Vcorrection = (float)(V[i][j] / (1 - Adam.B1_pow));
                    float Scorrection = (float)(S[i][j] / (1 - Adam.B2_pow));

                    WI[i][j] -= (float)(AgentClass.Study_rate * Vcorrection / (Math.Sqrt(Scorrection) + Adam.E));
                }
            }
            return x_out;
        }
        /// <summary>
        /// 初始化权值，np.random.randn(n) * sqrt(2.0/n)，遵循 sumInput 个数的正太分布
        /// </summary>
        /// <param name="sumInput">输入个数</param>
        /// <returns></returns>
        private float W_value_method(int sumInput)
        {
            float y = (float)r.NextDouble();
            float x = (float)r.NextDouble();
            float number = (float)(Math.Cos(2 * Math.PI * x) * Math.Sqrt(-2 * Math.Log(1 - y)));
            number *= (float)Math.Sqrt(2.0 / sumInput);
            return number;
        }
    }
}
