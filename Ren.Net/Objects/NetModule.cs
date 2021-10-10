using Ren.Net.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    [Serializable]
    public class NetModule
    {
        /// <summary>
        /// 优化器，用于优化 WI 更新
        /// </summary>
        public Optimizer Optimizer { set; get; }
        /// <summary>
        /// 用于初始化 WI
        /// </summary>
        public WIOptimizer WIOptimizer { set; get; }
        public virtual void Init()
        {
            
        }
        /// <summary>
        /// 输入 行是神经元的个数，列是 batchsize
        /// </summary>
        /// <param name="in"></param>
        /// <returns></returns>
        public virtual Torch Forward(Torch @in) 
        {
            throw new NotImplementedException();
        }

        public virtual Torch Backup(Torch @out)
        {
            throw new NotImplementedException();
        }
    }
}
