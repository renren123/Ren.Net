using Ren.Device;
using Ren.Net.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    [Serializable]
    public class NetModule : BaseNet
    {
        public virtual DeviceTpye Device { get; set; }
        public int MaxLinearNumber { set; get; }
        /// <summary>
        /// 优化器，用于优化 WI 更新
        /// </summary>
        public virtual Optimizer Optimizer { set; get; }
        /// <summary>
        /// 用于初始化 WI
        /// </summary>
        public virtual WIInitialization WIInitialize { set; get; }
        public virtual void Init()
        {
            
        }
        /// <summary>
        /// 输入 行是神经元的个数，列是 batchsize
        /// </summary>
        /// <param name="in"></param>
        /// <returns></returns>
        public virtual Tensor Forward(Tensor @in) 
        {
            throw new NotImplementedException();
        }

        public virtual Tensor Backup(Tensor @out)
        {
            throw new NotImplementedException();
        }
    }
}
