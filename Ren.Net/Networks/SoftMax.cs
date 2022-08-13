using Ren.Device;
using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Networks
{
    /// <summary>
    /// 参考：
    /// https://zhuanlan.zhihu.com/p/86184547
    /// https://zhuanlan.zhihu.com/p/37740860
    /// </summary>
    [Serializable]
    public class SoftMax : NetModule
    {
        private Tensor XOut { set; get; }
        public override void Init()
        {
            Log.Debug($"SoftMax inited");
        }
        public override Tensor Forward(Tensor @in)
        {
            for (int i = 0; i < @in.Row; i++)
            {
                for (int j = 0; j < @in.Column; j++)
                {
                    @in[i, j] = (float)Math.Exp(@in[i, j]);
                }
            }
            // 列的sum，为每个输出神经元总和求概率
            Tensor ePower = @in.Sum(axis: AxisType.Row);
            for (int i = 0; i < @in.Row; i++)
            {
                for (int j = 0; j < @in.Column; j++)
                {
                    @in[i, j] = @in[i, j] / ePower[i, 0];
                }
            }
            XOut = @in.Clone() as Tensor;
            return @in;
        }
        public override Tensor Backward(Tensor @out)
        {
            for (int i = 0; i < @out.Row; i++)
            {
                for (int j = 0; j < @out.Column; j++)
                {
                    @out[i, j] = 0F;
                }
            }

            for (int i = 0; i < @out.Row; i++)
            {
                for (int j = 0; j < @out.Column; j++)
                {
                    for (int k = 0; k < @out.Column; k++)
                    {
                        if (j == k)
                        {
                            @out[i, k] += XOut[i, k] - XOut[i, k] * XOut[i, k];
                        }
                        else
                        {
                            @out[i, k] += 0F - XOut[i, j] * XOut[i, k];
                        }
                    }
                }
            }

            return @out;
        }
    }
}
