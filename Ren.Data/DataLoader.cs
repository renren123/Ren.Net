using Ren.Net.Objects;
using System;
using System.Collections;

namespace Ren.Data
{
    public class DataLoader : IEnumerable
    {
        public virtual int Length { get => throw new NotImplementedException(); }
        public int BatchSize { set; get; }
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
            for (int i = 0; i < Length; i++)
            {
                yield return GetItem(i);
            }
        }
    }
}
