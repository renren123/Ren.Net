using Ren.Net.Objects;
using Ren.Net.Optimizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Networks
{
    public class BatchNorm1D : NetModule
    {
        public static bool IsTrain { set; get; } = true; 
        private static readonly float Momentum = 0.9F;
        /// <summary>
        /// 对应Ub
        /// </summary>
        private float RunningMean { set; get; }
        /// <summary>
        /// 对应Sigmab
        /// </summary>
        private float RunningVar { set; get; }
        private float[] x_hat { set; get; }
        private float UbAverage { set; get; }
        private float SigmmaAverage { set; get; }
        public float Ub { set; get; }
        public float Sigmab { set; get; }
        private float Gamma { set; get; } = 1;
        private float Bata { set; get; } = 1;

        public Adam AdamGamma { set; get; }
        public Adam AdamBata { set; get; }
        //public float Vgamma { set; get; }
        //public float Sgamma { set; get; }
        //public float Vbata { set; get; }
        //public float Sbata { set; get; }



        private Tensor X_IN { set; get; }
        private Tensor X_Hat { set; get; }
        /// <summary>
        /// 输入层神经元个数
        /// </summary>
        private int InputNumber { set; get; }
        private float[] UB { set; get; }
        private float[] SIGMAB { set; get; }
        private float[] RMean { set; get; }
        private float[] RVar { set; get; }

        private Tensor GammaBata { set; get; }

        public BatchNorm1D(int inputNumber)
        {
            this.InputNumber = inputNumber;
            GammaBata = new Tensor(inputNumber, 2, 1F);
        }

        public override void Init()
        {
            base.Init();
        }
        public override Tensor Forward(Tensor @in)
        {
            X_IN = @in.Clone() as Tensor;
            if (IsTrain)
            {
                X_Hat = new Tensor(@in.Row, @in.Column);
                UB = new float[InputNumber];
                SIGMAB = new float[InputNumber];
                float[] sigmabTemp = new float[InputNumber];

                if (RMean == null)
                {
                    RMean = new float[InputNumber];
                }
                if(RVar == null)
                {
                    RVar = new float[InputNumber];
                }

                for (int i = 0; i < UB.Length; i++)
                {
                    UB[i] = @in.ColumnAverage(i);
                    SIGMAB[i] = @in.ColumnVariance(i);
                    sigmabTemp[i] = (float)Math.Sqrt(SIGMAB[i] + Adam.E);
                }
                for (int i = 0; i < @in.Row; i++)
                {
                    for (int j = 0; j < @in.Column; j++)
                    {
                        X_Hat[i, j] = (@in[i, j] - UB[i]) / sigmabTemp[i];
                        @in[i, j] = GammaBata[i, 0] * X_Hat[i, j] + GammaBata[i, 1];
                    }
                }
                return @in;
            }
            else
            {
                float[] scale = new float[InputNumber];
                for (int i = 0; i < InputNumber; i++)
                {
                    scale[i] = GammaBata[i, 0] / (float)Math.Sqrt(RVar[i] + Adam.E);
                    @in[i, 0] = @in[i, 0] * scale[i] + (GammaBata[i, 1] - RMean[i] * scale[i]);
                }
            }
            return @in;
        }
        public override Tensor Backup(Tensor @out)
        {
            int batchSize = @out.Column;
            // 处理 一个 batch
            Tensor dx_hat = new Tensor(@out.Row, @out.Column);
            Tensor dx = new Tensor(@out.Row, @out.Column);
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
                    sum += dx_hat[i, j] * (X_IN[i, j] - UB[i]);
                }
                dx_hat_sum[i] = sum;
            }
            for (int i = 0; i < dsigmab.Length; i++)
            {
                dsigmab[i] = dx_hat_sum[i] * (-0.5F) * (float)Math.Pow((SIGMAB[i] + Adam.E), -1.5);
            }
            float[] dub_temp_1 = new float[InputNumber];
            for (int i = 0; i < InputNumber; i++)
            {
                float sum = 0F;
                for (int j = 0; j < @out.Column; j++)
                {
                    sum += (float)(dx_hat[i, j] * ((-1) / Math.Sqrt(SIGMAB[i] + Adam.E)));
                }
                dub_temp_1[i] = sum;
            }
            float[] dub_temp_2 = new float[InputNumber];
            for (int i = 0; i < InputNumber; i++)
            {
                float sum = 0F;
                for (int j = 0; j < @out.Column; j++)
                {
                    sum += (-2) * (X_IN[i, j] - Ub);
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
                    dx[i, j] = (float)(dx_hat[i, j] / Math.Sqrt(Sigmab + Adam.E) + dsigmab[i] * 2 * (X_IN[i, j] - UB[i]) / batchSize + dub[i] / batchSize);
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

            GammaBata -= Optimizer.GetOptimizer(dgammaBata);
            return dx;
        }
        //得到输出
        /// <summary>
        /// 前向传播
        /// </summary>
        /// <param name="xi"></param>
        /// <returns></returns>
        public float[] GetX_out(float[] xi)
        {
            float[] x_out = new float[xi.Length];
            if (IsTrain)
            {
                x_hat = new float[xi.Length];
                Ub = xi.Average();
                // *********************************************
                // Sigmab = ArrayAction.variance(xi, Ub);
                // *********************************************
                RunningMean = Momentum * RunningMean + (1 - Momentum) * Ub;
                RunningVar = Momentum * RunningVar + (1 - Momentum) * Sigmab;
                float sigmabTemp = (float)Math.Sqrt(Sigmab + Adam.E);
                for (int i = 0; i < x_hat.Length; i++)
                {
                    x_hat[i] = (xi[i] - Ub) / sigmabTemp;
                    x_out[i] = Gamma * x_hat[i] + Bata;
                }
                //if (ClassPublicValue.isReadyToUbAndSigmma == true)
                //{
                //    int countNumber = BNStep;//当前的数量
                //    UbAverage = (UbAverage * (countNumber - 1) + Ub) / countNumber;
                //    SigmmaAverage = (SigmmaAverage * (countNumber - 1) + Sigmab) / countNumber;
                //}
                return x_out;
            }
            else
            {
                //scale = gamma / np.sqrt(running_var + eps)
                double scale = Gamma / Math.Sqrt(RunningVar + Adam.E);
                x_out[0] = (float)(xi[0] * scale + (Bata - RunningMean * scale));
                return x_out;

                //int Mcount = AgentClass.Mini_batchSizeTrain;
                ////float xishu = 1;
                //float UbTemp = UbAverage;
                //float SigmmaTemp = SigmmaAverage * Mcount/(Mcount-1);
                //float x_hatTest = (float)((xi[0] - UbTemp) / Math.Sqrt(SigmmaTemp + Adam.E));
                //x_out[0] = Gamma * x_hatTest + Bata;
                //// x_out[0] = (Gamma / (Math.Sqrt(Sigmabtrain + Adam.E))) * xi[0] + (Bata - Gamma * Ubtrain / Math.Sqrt(Sigmabtrain + Adam.E));

                //return x_out ;
            }
            return null;
        }
        private float GetListAverage(List<float> list)
        {
            float sum = 0;
            for (int i = 0; i < list.Count; i++)
            {
                sum += list[i];
            }
            return sum / list.Count;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="delta_yi"></param>
        /// <returns></returns>
        public float[] GetDout(float[] delta_yi, float[] xi)
        {
            float[] dx_hat = new float[delta_yi.Length];
            float dsigmab = 0;
            float dub = 0;
            float dgamma = 0;
            float dbata = 0;
            float[] dx = new float[delta_yi.Length];
            for (int i = 0; i < dx_hat.Length; i++)
            {
                dx_hat[i] = delta_yi[i] * Gamma;
            }
            float dx_hat_sum = 0;
            for (int i = 0; i < delta_yi.Length; i++)
            {
                dx_hat_sum += dx_hat[i] * (xi[i] - Ub);
            }

            dsigmab = (float)(dx_hat_sum * (-0.5) * Math.Pow((Sigmab + Adam.E), -1.5));

            float dub_temp_1 = 0;
            for (int i = 0; i < dx_hat.Length; i++)
            {
                dub_temp_1 += (float)(dx_hat[i] * ((-1) / Math.Sqrt(Sigmab + Adam.E)));
            }
            float dub_temp_2 = 0;
            for (int i = 0; i < delta_yi.Length; i++)
            {
                dub_temp_2 += (-2) * (xi[i] - Ub);
            }
            dub = dub_temp_1 + dsigmab * dub_temp_2 / delta_yi.Length;

            for (int i = 0; i < dx.Length; i++)
            {
                dx[i] = (float)(dx_hat[i] / Math.Sqrt(Sigmab + Adam.E) + dsigmab * 2 * (xi[i] - Ub) / dx.Length + dub / dx.Length);
            }
            for (int i = 0; i < dx_hat.Length; i++)
            {
                dgamma += delta_yi[i] * x_hat[i];
                dbata += delta_yi[i];
            }
            //更新


            // *********************************************
            //if (AdamGamma == null)
            //    AdamGamma = new Adam();
            //if (AdamBata == null)
            //    AdamBata = new Adam();
            //Gamma -= AdamGamma.GetAdam(dgamma);
            //Bata -= AdamBata.GetAdam(bata);
            // *********************************************
            return dx;
        }
    }
}
