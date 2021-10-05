using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Fully.Network
{
    /// <summary>
    /// 一层神经元
    /// </summary>
    class NetLayer
    {
        /// <summary>
        /// 一层神经元 存储的结构
        /// </summary>
        public List<NetNeuron> FullyNeurns { set; get; }
    }
}
