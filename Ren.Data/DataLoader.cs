using Ren.Net.Objects;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Ren.Net.Extensions;

namespace Ren.Data
{
    public class DataLoader : IEnumerable
    {
        public virtual int Length { get => throw new NotImplementedException(); }
        public int BatchSize { set; get; }
        /// <summary>
        /// 表示是否打乱数据
        /// </summary>
        public bool Shuffle { set; get; }

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
        public IEnumerator GetEnumerator()
        {
            if (!this.Shuffle)
            {
                for (int i = 0; i < Length; i++)
                {
                    yield return GetItem(i);
                }
            }
            else
            {
                // 左闭右开 => [0, Length)
                List<int> range = Enumerable.Range(0, Length).ToList();
                range.ListRandom();
                foreach (var index in range)
                {
                    yield return GetItem(index);
                }
            }
        }
    }
}
