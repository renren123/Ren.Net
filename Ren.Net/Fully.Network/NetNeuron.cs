using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Fully.Network
{
    /// <summary>
    /// 单个神经元
    /// </summary>
    public class NetNeuron
    {
        /// <summary>
        /// 权重，单个神经元的权重数量是上一层神经元的个数，每个元素 都是上一层神经元 到自己
        /// 神经元的权重
        /// </summary>
        public float[] Wi { set; get; }
        /// <summary>
        /// 误差值，与本层神经元的数量相同
        /// </summary>
        public float[] BehindWi { set; get; }
    }
}
