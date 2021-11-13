using Ren.Device;
using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Networks
{
    [Serializable]
    public class BatchNorm1DCPU : BatchNorm1D
    {
        public override DeviceTpye Device { get => DeviceTpye.CPU; }

        public BatchNorm1DCPU(int inputNumber) : base(inputNumber)
        {
        }
        public override void Init()
        {
            GammaBata = new Tensor(InputNumber, 2, 1F);
            UbSigmaB = new Tensor(InputNumber, 2, 0F);
            RMeanVar = new Tensor(InputNumber, 2, 0F);
        }
        public override Tensor Forward(Tensor @in)
        {
            X_IN = @in.Clone() as Tensor;
            if (IsTrain)
            {
                if (X_Hat == null)
                {
                    X_Hat = new Tensor(@in.Row, @in.Column, 0F);
                }
                for (int i = 0; i < InputNumber; i++)
                {
                    UbSigmaB[i, 0] = @in.RowAverage(i);
                    UbSigmaB[i, 1] = @in.RowVariance(i);
                }
                for (int i = 0; i < InputNumber; i++)
                {
                    RMeanVar[i, 0] = Momentum * RMeanVar[i, 0] + (1 - Momentum) * UbSigmaB[i, 0];
                    RMeanVar[i, 1] = Momentum * RMeanVar[i, 1] + (1 - Momentum) * UbSigmaB[i, 1];
                }
                for (int i = 0; i < @in.Row; i++)
                {
                    for (int j = 0; j < @in.Column; j++)
                    {
                        X_Hat[i, j] = (@in[i, j] - UbSigmaB[i, 0]) / (float)Math.Sqrt(UbSigmaB[i, 1] + E);
                        @in[i, j] = GammaBata[i, 0] * X_Hat[i, j] + GammaBata[i, 1];
                    }
                }
            }
            else
            {
                float[] scale = new float[InputNumber];
                for (int i = 0; i < InputNumber; i++)
                {
                    scale[i] = GammaBata[i, 0] / (float)Math.Sqrt(RVar[i] + E);
                    @in[i, 0] = @in[i, 0] * scale[i] + (GammaBata[i, 1] - RMean[i] * scale[i]);
                }
            }
            return @in;
        }
        public override Tensor Backup(Tensor @out)
        {
            int batchSize = @out.Column;
            // 处理 一个 batch
            Tensor dx_hat = new Tensor(@out.Row, @out.Column, 0F);
            Tensor dx = new Tensor(@out.Row, @out.Column, 0F);
            float[] dsigmab = new float[InputNumber];
            float[] dub = new float[InputNumber];
            // 第一行 是 gamma  第二行 Bata
            Tensor dgammaBata = new Tensor(InputNumber, 2, 0F);

            //float[] dgamma = new float[InputNumber];
            //float[] dbata = new float[InputNumber];
            float[] dx_hat_sum = new float[InputNumber];

            for (int i = 0; i < @out.Row; i++)
            {
                for (int j = 0; j < @out.Column; j++)
                {
                    dx_hat[i, j] = @out[i, j] * GammaBata[i, 0];
                }
            }

            for (int i = 0; i < InputNumber; i++)
            {
                float sum = 0F;
                for (int j = 0; j < @out.Column; j++)
                {
                    sum += dx_hat[i, j] * (X_IN[i, j] - UbSigmaB[i, 0]);
                }
                dx_hat_sum[i] = sum;
            }

            for (int i = 0; i < dsigmab.Length; i++)
            {
                dsigmab[i] = dx_hat_sum[i] * (-0.5F) * (float)Math.Pow((UbSigmaB[i, 1] + E), -1.5);
            }

            float[] dub_temp_1 = new float[InputNumber];
            for (int i = 0; i < InputNumber; i++)
            {
                float sum = 0F;
                for (int j = 0; j < @out.Column; j++)
                {
                    sum += (float)(dx_hat[i, j] * ((-1) / Math.Sqrt(UbSigmaB[i, 1] + E)));
                }
                dub_temp_1[i] = sum;
            }
            float[] dub_temp_2 = new float[InputNumber];
            for (int i = 0; i < InputNumber; i++)
            {
                float sum = 0F;
                for (int j = 0; j < @out.Column; j++)
                {
                    sum += (-2) * (X_IN[i, j] - UbSigmaB[i, 0]);
                }
                dub_temp_2[i] = sum;
            }
            for (int i = 0; i < InputNumber; i++)
            {
                dub[i] = dub_temp_1[i] + dsigmab[i] * dub_temp_2[i] / batchSize;
            }
            for (int i = 0; i < InputNumber; i++)
            {
                for (int j = 0; j < @out.Column; j++)
                {
                    dx[i, j] = (float)(dx_hat[i, j] / Math.Sqrt(UbSigmaB[i, 1] + E) + dsigmab[i] * 2 * (X_IN[i, j] - UbSigmaB[i, 0]) / batchSize + dub[i] / batchSize);
                }
            }
            for (int i = 0; i < InputNumber; i++)
            {
                for (int j = 0; j < @out.Column; j++)
                {
                    dgammaBata[i, 0] += @out[i, j] * X_Hat[i, j];
                    dgammaBata[i, 1] += @out[i, j];
                }
            }
            GammaBata -= Optimizer.GetOptimizer(dgammaBata, null);
            return dx;
        }
    }
}
