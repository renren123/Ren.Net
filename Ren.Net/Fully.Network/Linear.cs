using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Fully.Network
{
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
        // public int BatchSize { set; get; }
        /// <summary>
        /// 一层神经元 存储的结构
        /// </summary>
        public List<NetNeuron> FullyNeurns { set; get; }
        /// <summary>
        /// 权重单独保存在一个地图里面，方向是正向传播的方向，list 每个元素是当前 神经元素的个数，float[] 数组是上一层元素的个数
        /// </summary>
        public List<float[]> WI { set; get; }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputSize"></param>
        /// <param name="outputSize">当前层 神经元的数量</param>
        public Linear(int inputNumber, int outputNumber)
        {
            this.InputNumber = inputNumber;
            this.OutputNumber = outputNumber;

            FullyNeurns = new List<NetNeuron>(OutputNumber);
            for (int i = 0; i < OutputNumber; i++)
            {
                FullyNeurns.Add(new NetNeuron()
                {
                    Wi = new float[OutputNumber],
                    //SensitiveValue = new float[outputSize]  // 这个是不是要重新初始化
                });
            }
            // ******************************* 初始化 wi *******************************
        }
        public override Torch Forward(Torch @in)
        {
            int batchSize = @in.BatchSize;          // batch 的大小

            if (batchSize == -1)
            {
                throw new Exception("Linear::Forward, batchSize is -1 or neuronNumber is -1");
            }
            Torch x_out = new Torch(OutputNumber, batchSize);
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
            return x_out;
        }
    }
}
