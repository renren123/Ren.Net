using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Networks
{
    //https://zhuanlan.zhihu.com/p/86184547
    [Serializable]
    public class SoftMax : NetModule
    {
        private Tensor XIn { set; get; }
        public override void Init()
        {
            Log.Debug($"SoftMax inited");
        }
        public override Tensor Forward(Tensor @in)
        {
            XIn = @in.Clone() as Tensor;

            for (int i = 0; i < @in.Row; i++)
            {
                for (int j = 0; j < @in.Column; j++)
                {
                    @in[i, j] = (float)Math.Exp(@in[i, j]);
                }
            }
            // 列的sum，为每个输出神经元总和求概率
            Tensor ePower = @in.Sum(axis: 0);
            for (int i = 0; i < @in.Row; i++)
            {
                for (int j = 0; j < @in.Column; j++)
                {
                    @in[i, j] = @in[i, j] / ePower[i, 0];
                }
            }
            return @in;
        }
        public override Tensor Backup(Tensor @out)
        {
            return base.Backup(@out);
        }
    }
}
