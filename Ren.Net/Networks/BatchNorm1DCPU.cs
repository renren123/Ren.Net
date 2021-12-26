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

            for (int i = 0; i < InputNumber; i++)
            {
                GammaBata[i, 1] = 0F;
            }
            UbSigmaB = new Tensor(InputNumber, 2, 0F);
            RMeanVar = new Tensor(InputNumber, 2, 0F);

            UB = new Tensor(InputNumber, 1, 0F);
            SigmaB = new Tensor(InputNumber, 1, 0F);
            RMean = new Tensor(InputNumber, 1, 0F);
            RVar = new Tensor(InputNumber, 1, 0F);
            Gamma = new Tensor(InputNumber, 1, 1F);
            Bata = new Tensor(InputNumber, 1, 0F);
        }
        public override Tensor Forward(Tensor @in)
        {
            X_IN = @in.Clone() as Tensor;

            if (IsTrain)
            {
                // numpy使用之np.matmul https://blog.csdn.net/alwaysyxl/article/details/83050137
                // Batch Normalization及其反向传播 https://zhuanlan.zhihu.com/p/45614576

                //X_IN = @in.Clone() as Tensor;

                UB = @in.Mean(1);
                SigmaB = @in.Variance(1);

                //PrintArray(@in.ToArray());

                X_Hat = Tensor.DotDivide(@in - UB, (SigmaB + E).Sqrt());

                @in = Tensor.DotMultiply(X_Hat, Gamma) + Bata;

                RMean = Momentum * RMean + (1 - Momentum) * UB;
                RVar = Momentum * RVar + (1 - Momentum) * SigmaB;
            }
            else
            {

            }
            return @in;

            // ################ OLD #######################

            //X_IN = @in.Clone() as Tensor;
            //if (IsTrain)
            //{
            //    if (X_Hat == null)
            //    {
            //        X_Hat = new Tensor(@in.Row, @in.Column, 0F);
            //    }
            //    for (int i = 0; i < InputNumber; i++)
            //    {
            //        UbSigmaB[i, 0] = @in.RowAverage(i);
            //        UbSigmaB[i, 1] = @in.RowVariance(i);
            //    }
            //    for (int i = 0; i < InputNumber; i++)
            //    {
            //        RMeanVar[i, 0] = Momentum * RMeanVar[i, 0] + (1 - Momentum) * UbSigmaB[i, 0];
            //        RMeanVar[i, 1] = Momentum * RMeanVar[i, 1] + (1 - Momentum) * UbSigmaB[i, 1];
            //    }
            //    // PrintArray(UbSigmaB.ToArray());
            //    // PrintArray(RMeanVar.ToArray());
            //    for (int i = 0; i < @in.Row; i++)
            //    {
            //        for (int j = 0; j < @in.Column; j++)
            //        {
            //            X_Hat[i, j] = (@in[i, j] - UbSigmaB[i, 0]) / (float)Math.Sqrt(UbSigmaB[i, 1] + E);
            //            @in[i, j] = GammaBata[i, 0] * X_Hat[i, j] + GammaBata[i, 1];
            //        }
            //    }
            //    //PrintArray(X_Hat.ToArray());
            //    //PrintArray(@in.ToArray());
            //}
            //else
            //{
            //    float[] scale = new float[InputNumber];
            //    for (int i = 0; i < InputNumber; i++)
            //    {
            //        scale[i] = GammaBata[i, 0] / (float)Math.Sqrt(RMeanVar[i, 1] + E);
            //        @in[i, 0] = @in[i, 0] * scale[i] + (GammaBata[i, 1] - RMeanVar[i, 0] * scale[i]);
            //    }
            //}
            //return @in;

        }
        public override Tensor Backup(Tensor @out)
        {
            //int N = @out.Column;

            //Tensor dbeta = @out.Sum(axis: 1);
            //Tensor dgama = Tensor.DotMultiply(X_Hat, @out).Sum(axis: 1);

            //Tensor dxhat = Tensor.DotMultiply(@out, Gamma);
            //Tensor dvar = Tensor.DotMultiply(Tensor.DotMultiply(dxhat, X_Hat).Sum(axis: 1), (-0.5F) * (1.0F / SigmaB + E));
            //Tensor dmean = Tensor.DotDivide(dxhat.Sum(axis: 1), (SigmaB + E).Sqrt()) + Tensor.DotMultiply(dvar * (-2F), UB);

            //Tensor dx = Tensor.DotDivide(dxhat, (SigmaB + E).Sqrt()) + Tensor.DotMultiply((X_IN - UB) / N, dvar * 2) + dmean / N;

            //Gamma -= 0.001F * dgama;
            //Bata -= 0.001F * dbeta;
            //return dx;


            int N = @out.Column;

            Tensor var_plus_eps = SigmaB + E;
            Tensor dGamma = Tensor.DotMultiply(X_Hat, @out).Sum(axis: 1);
            Tensor dBeta = @out.Sum(axis: 1);

            Tensor ones = new Tensor(1, N, 1F);
            Tensor dx_ = Tensor.DotMultiply(Gamma * ones, @out);

            Tensor dx = N * dx_ - dx_.Sum(axis: 1) - Tensor.DotMultiply(X_Hat, Tensor.DotMultiply(dx_, X_Hat).Sum(axis: 1));
            Tensor dx_out = Tensor.DotDivide(dx * (1.0F / N), var_plus_eps.Sqrt());

            Console.WriteLine("dGamma");
            PrintArray(dGamma.ToArray());
            Console.WriteLine("dBeta");
            PrintArray(dBeta.ToArray());

            Gamma -= 0.1F * dGamma;
            Bata -= 0.1F * dBeta;

            //Console.WriteLine("Gamma");
            //PrintArray(Gamma.ToArray());
            //Console.WriteLine("Bata");
            //PrintArray(Bata.ToArray());
            return dx_out;


            // ################ OLD #######################

            //int batchSize = @out.Column;
            //// 处理 一个 batch
            //Tensor dx_hat = new Tensor(@out.Row, @out.Column, 0F);
            //Tensor dx = new Tensor(@out.Row, @out.Column, 0F);
            //float[] dsigmab = new float[InputNumber];
            //float[] dub = new float[InputNumber];
            //// 第一行 是 gamma  第二行 Bata
            //Tensor dgammaBata = new Tensor(InputNumber, 2, 0F);

            //float[] dx_hat_sum = new float[InputNumber];

            //for (int i = 0; i < @out.Row; i++)
            //{
            //    for (int j = 0; j < @out.Column; j++)
            //    {
            //        dx_hat[i, j] = @out[i, j] * GammaBata[i, 0];
            //    }
            //}

            //for (int i = 0; i < InputNumber; i++)
            //{
            //    float sum = 0F;
            //    for (int j = 0; j < @out.Column; j++)
            //    {
            //        sum += dx_hat[i, j] * (X_IN[i, j] - UbSigmaB[i, 0]);
            //    }
            //    dx_hat_sum[i] = sum;
            //}

            //for (int i = 0; i < dsigmab.Length; i++)
            //{
            //    dsigmab[i] = dx_hat_sum[i] * (-0.5F) * (float)Math.Pow((UbSigmaB[i, 1] + E), -1.5);
            //}

            //float[] dub_temp_1 = new float[InputNumber];
            //for (int i = 0; i < InputNumber; i++)
            //{
            //    float sum = 0F;
            //    for (int j = 0; j < @out.Column; j++)
            //    {
            //        sum += (float)(dx_hat[i, j] * ((-1) / Math.Sqrt(UbSigmaB[i, 1] + E)));
            //    }
            //    dub_temp_1[i] = sum;
            //}
            //float[] dub_temp_2 = new float[InputNumber];
            //for (int i = 0; i < InputNumber; i++)
            //{
            //    float sum = 0F;
            //    for (int j = 0; j < @out.Column; j++)
            //    {
            //        sum += (-2) * (X_IN[i, j] - UbSigmaB[i, 0]);
            //    }
            //    dub_temp_2[i] = sum;
            //}
            //for (int i = 0; i < InputNumber; i++)
            //{
            //    dub[i] = dub_temp_1[i] + dsigmab[i] * dub_temp_2[i] / batchSize;
            //}

            //for (int i = 0; i < InputNumber; i++)
            //{
            //    for (int j = 0; j < @out.Column; j++)
            //    {
            //        dx[i, j] = (float)(dx_hat[i, j] / Math.Sqrt(UbSigmaB[i, 1] + E) + dsigmab[i] * 2 * (X_IN[i, j] - UbSigmaB[i, 0]) / batchSize + dub[i] / batchSize);
            //    }
            //}
            //for (int i = 0; i < InputNumber; i++)
            //{
            //    for (int j = 0; j < @out.Column; j++)
            //    {
            //        dgammaBata[i, 0] += @out[i, j] * X_Hat[i, j];
            //        dgammaBata[i, 1] += @out[i, j];
            //    }
            //}

            ////for (int i = 0; i < InputNumber; i++)
            ////{
            ////    for (int j = 0; j < 2; j++)
            ////    {
            ////        GammaBata[i, j] -= dgammaBata[i, j];
            ////    }
            ////}

            ////PrintArray(dgammaBata.ToArray());

            //GammaBata -= Optimizer.GetOptimizer(dgammaBata, null);
            //return dx;

        }
    }
}
