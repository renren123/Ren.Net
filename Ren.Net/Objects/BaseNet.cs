using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    [Serializable]
    public class BaseNet
    {
        public NetParameter NetParameter { set; get; }
        public virtual void LoadNetParameter(NetParameter netParameter)
        {
            this.NetParameter = netParameter;
        }
        public virtual void SetPublicValue(params Tensor[] datas)
        {
            foreach (var item in datas)
            {
                NetParameter.SetRegisterParameter(item);
            }
        }
    }
}
