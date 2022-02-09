using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Objects
{
    public class NetParameter
    {
        private int RegisterListCount { get; } = 5;
        private ConcurrentDictionary<Tensor, bool> RegisterDic { set; get; } = new ConcurrentDictionary<Tensor, bool>();

        public void Init(int maxLinearNumber)
        {
            for (int i = 0; i < RegisterListCount; i++)
            {
                Tensor data = new Tensor(maxLinearNumber, maxLinearNumber, 0F);
                RegisterDic[data] = false;
            }
        }
        public Tensor GetRegisterParameter()
        {
            foreach (var item in RegisterDic.Reverse())
            {
                if (!item.Value)
                {
                    RegisterDic[item.Key] = true;
                    return item.Key;
                }
            }
            return null;
        }
        public void SetRegisterParameter(Tensor data)
        {
            RegisterDic[data] = false;
        }
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }
    public class NetRegisterParameter
    {
        public Tensor Data { set; get; }
        public bool Sign { set; get; }
    }
}
