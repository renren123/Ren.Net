using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Fully.Network
{
    class BNElement
    {
        private float momentum = 0.9F;
        ///// <summary>
        ///// 这个数值是为了调整全连接层与卷积层学习速率的问题
        ///// </summary>
        //private double adjustBNNumber = 1;
        /// <summary>
        /// 对应Ub
        /// </summary>
        private float running_mean { set; get; }
        /// <summary>
        /// 对应Sigmab
        /// </summary>
        private float running_var { set; get; }
        private float[] x_hat;
        private float UbAverage { set; get; }
        private float SigmmaAverage { set; get; }
        public float Ub { set; get; }
        public float Sigmab { set; get; }
        float gamma = 1;
        float bata = 1;
        public Adam AdamGamma { set; get; }
        public Adam AdamBata { set; get; }
        //public float Vgamma { set; get; }
        //public float Sgamma { set; get; }
        //public float Vbata { set; get; }
        //public float Sbata { set; get; }

        public BNElement()
        {
            //Vgamma = Sgamma = Vbata = Sbata = 0;
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
            if (ClassPublicValue.TrainOrTest.Equals("train"))
            {
                x_hat = new float[xi.Length];
                Ub = ArrayAction.average(xi);
                Sigmab = ArrayAction.variance(xi, Ub);
                running_mean = momentum * running_mean + (1 - momentum) * Ub;
                running_var = momentum * running_var + (1 - momentum) * Sigmab;
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
            else if (ClassPublicValue.TrainOrTest.Equals("test"))
            {
                //scale = gamma / np.sqrt(running_var + eps)
                double scale = gamma / Math.Sqrt(running_var + Adam.E);
                x_out[0] = (float)(xi[0] * scale + (bata - running_mean * scale));
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

            if (AdamGamma == null)
                AdamGamma = new Adam();
            if (AdamBata == null)
                AdamBata = new Adam();
            Gamma -= AdamGamma.GetAdam(dgamma);
            Bata -= AdamBata.GetAdam(bata);
            //BNAdam(dgamma, dbata);
            return dx;
        }
        //private void BNAdam(float dgamma,float dbata)
        //{
        //    Vgamma = (float)(AgentClass.B1 * Vgamma + (1 - AgentClass.B1) * dgamma);
        //    Vbata= (float)(AgentClass.B1 * Vbata + (1 - AgentClass.B1) * dbata);
        //    Sgamma = (float)(AgentClass.B2 * Sgamma + (1 - AgentClass.B2) * dgamma * dgamma);
        //    Sbata = (float)(AgentClass.B2 * Sbata + (1 - AgentClass.B2) * dbata * dbata);
        //    float Vgamma_correction = (float)(Vgamma / (1 - Adam.B1_pow));
        //    float Vbata_correction = (float)(Vbata / (1 - Adam.B1_pow));
        //    float Sgamma_correction = (float)(Sgamma / (1 - Adam.B2_pow));
        //    float Sbata_correction = (float)(Sbata / (1 - Adam.B2_pow));

        //    Gamma -= (float)(adjustBNNumber * AgentClass.Study_rate * Vgamma_correction / (Math.Sqrt(Sgamma_correction) + Adam.E));
        //    Bata -= (float)(adjustBNNumber *AgentClass.Study_rate * Vbata_correction / (Math.Sqrt(Sbata_correction) + Adam.E));
        //}


        public float Gamma
        {
            get
            {
                return gamma;
            }

            set
            {
                gamma = value;
            }
        }

        public float Bata
        {
            get
            {
                return bata;
            }

            set
            {
                bata = value;
            }
        }
    }
}
