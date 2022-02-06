using Ren.Net.Objects;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Ren.Net.Extensions;

namespace Ren.Data
{
    /// <summary>
    /// 如果最后一个batchsize 不够一个 batch，随机填充
    /// </summary>
    public class DataLoader : IEnumerable
    {
        public virtual int Length { get => throw new NotImplementedException(); }
        public int BatchSize { set; get; } = 1;
        /// <summary>
        /// 表示是否打乱数据
        /// </summary>
        public bool Shuffle { set; get; } = false;

        private Random RD { set; get; } = new Random(System.Guid.NewGuid().GetHashCode());

        public DataLoader(int batchSize = 1, bool shuffle = false)
        {
            this.BatchSize = batchSize;
            this.Shuffle = shuffle;
        }
        public virtual void Init()
        {
            throw new NotImplementedException();
        }

        public virtual (Tensor data, Tensor label) GetItem(int index)
        {
            throw new NotImplementedException();
        }
        private (Tensor data, Tensor label) MergeItem(List<int> batchArray)
        {
            int inputNumber;
            int outputNumber;
            Tensor datas = null;
            Tensor labels = null;

            for (int i = 0; i < batchArray.Count; i++)
            {
                (Tensor data, Tensor label) = GetItem(batchArray[i]);

                inputNumber = data.Width;
                outputNumber = label.Width;

                if (datas == null)
                {
                    datas = new Tensor(inputNumber, BatchSize, 0F);
                }
                if (labels == null)
                {
                    labels = new Tensor(outputNumber, BatchSize, 0F);
                }
                // 列遍历是 神经元个数， 行遍历是 batch
                // data 是先第一列遍历
                for (int j = 0; j < inputNumber; j++)
                {
                    datas[j, i] = data[j];
                }
                for (int j = 0; j < outputNumber; j++)
                {
                    labels[j, i] = label[j];
                }
            }
            return (datas, labels);
        }
        public IEnumerator GetEnumerator()
        {
            if (Length == 0)
            {
                throw new Exception($"GetEnumerator length is {Length}");
            }
            // 左闭右开 => [0, Length)
            List<int> range = Enumerable.Range(0, Length).ToList();

            if (this.Shuffle)
            {
                range.ListRandom();
            }

            foreach (var item in range.SplitList(BatchSize))
            {
                // 说明是最后一个，随机填充数据到 一个 BatchSize
                if (item.Count != BatchSize)
                {
                    while (item.Count < BatchSize)
                    {
                        item.Add(RD.Next(1, Length));
                    }
                }
                yield return MergeItem(item);
            }
        }
    }
}
