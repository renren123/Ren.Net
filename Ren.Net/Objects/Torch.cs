using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    public class Torch
    {
        /// <summary>
        /// 几个神经元 batch 数据，list 的长度 是一层神经元的数量，float 是 batch 的大小
        /// </summary>
        public List<float[]> Data { set; get; }
        public int BatchSize => Data == null || Data.Count == 0 ? -1 : Data[0].Length;
        public int LastNeuronNumber => Data == null || Data.Count == 0 ? -1 : Data.Count;
        public Torch(List<float[]> data)
        {
            this.Data = new List<float[]>(data);
        }
        /// <summary>
        /// 初始化 一个 torch
        /// </summary>
        /// <param name="neuronNumber">神经元的数量</param>
        /// <param name="batch">一个 batch 的大小</param>
        public Torch(int neuronNumber, int batch)
        {
            this.Data = new List<float[]>(neuronNumber);
            for (int i = 0; i < neuronNumber; i++)
            {
                this.Data.Add(new float[batch]);
            }
        }
    }
}
