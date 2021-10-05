using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Fully.Network
{
    public class Linear : NetModule
    {
        /// <summary>
        /// 输入层神经元个数
        /// </summary>
        public int InputSize { set; get; }
        /// <summary>
        /// 输出层神经元个数
        /// </summary>
        public int OutputSize { set; get; }
        /// <summary>
        /// 第一层神经元
        /// </summary>
        private NetLayer firstLayer { set; get; }
        public Linear(int inputSize, int outputSize)
        {
            this.InputSize = inputSize;
            this.OutputSize = outputSize;
        }

    }
}
