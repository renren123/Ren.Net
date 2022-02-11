using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Objects
{
    [Serializable]
    public class NetParameter
    {
        public int RegisterListCount { set; get; }
        private readonly object registerDicLock = new object();
        private Dictionary<Tensor, bool> RegisterDic { set; get; } = new Dictionary<Tensor, bool>();
        public void Init(int maxLinearNumber, int registerListCount)
        {
            if (maxLinearNumber == 0 || registerListCount == 0)
            {
                throw new Exception($"maxLinearNumber or registerListCount is zero, maxLinearNumber: {maxLinearNumber}, registerListCount: {registerListCount}");
            }
            this.RegisterListCount = registerListCount;
            for (int i = 0; i < RegisterListCount; i++)
            {
                Tensor data = new Tensor(maxLinearNumber, maxLinearNumber, 0F);
                RegisterDic[data] = false;
            }
        }
        public Tensor GetRegisterParameter()
        {
            lock (registerDicLock)
            {
                foreach (var item in RegisterDic)
                {
                    if (!item.Value)
                    {
                        RegisterDic[item.Key] = true;
                        return item.Key;
                    }
                }
            }
            return null;
        }
        public void SetRegisterParameter(Tensor data)
        {
            lock (registerDicLock)
            {
                if (data == null || !RegisterDic.ContainsKey(data) || !RegisterDic[data])
                {
                    throw new Exception("data check error");
                }
                RegisterDic[data] = false;
            } 
        }
    }
}
