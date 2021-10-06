using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    public class Torch : ICloneable
    {
        /// <summary>
        /// 几个神经元 batch 数据，list 的长度 是一层神经元的数量，float 是 batch 的大小
        /// </summary>
        public List<float[]> Data { set; get; }
        public int BatchSize => Data == null || Data.Count == 0 ? -1 : Data[0].Length;
        public int LastNeuronNumber => Data == null || Data.Count == 0 ? -1 : Data.Count;
        public Torch() { }
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

        public object Clone()
        {
            Torch torch = new Torch()
            {
                Data = new List<float[]>(this.Data.Count)
            };
            foreach (var item in this.Data)
            {
                float[] temp = new float[item.Length];

                for (int i = 0; i < item.Length; i++)
                {
                    temp[i] = item[i];
                }
                torch.Data.Add(temp);
            }
            return torch;
        }
    }
}
