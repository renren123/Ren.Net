using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Fully.Network
{
    class Fullyneuron
    {
        /// <summary>
        /// 这个数值是为了调整全连接层与卷积层学习速率的问题
        /// </summary>
        private double adjustWiNumber = 1;
        public BNElement BN { set; get; }
        public float[] Xout { set; get; }
        public float[] BN_out { set; get; }
        public float[] Xin { set; get; }
        /// <summary>
        /// 注意这个SensitiveValue是在BN前的敏感度，是用于更新Wi权值的敏感度
        /// </summary>
        public float[] SensitiveValue { set; get; }
        public float[] Wi { set; get; }
        public float[] V { set; get; }
        public float[] S { set; get; }
        public Fullyneuron()
        {
            BN = new BNElement();
        }
        public void AdamUpdateWi(float dw, int index)
        {
            V[index] = (float)(AgentClass.B1 * V[index] + (1 - AgentClass.B1) * dw);
            S[index] = (float)(AgentClass.B2 * S[index] + (1 - AgentClass.B2) * dw * dw);
            float Vcorrection = (float)(V[index] / (1 - Adam.B1_pow));
            float Scorrection = (float)(S[index] / (1 - Adam.B2_pow));
            Wi[index] -= (float)(adjustWiNumber * AgentClass.Study_rate * Vcorrection / (Math.Sqrt(Scorrection) + Adam.E));
        }
        private float[] F_back(float[] x_Sout)
        {
            for (int i = 0; i < x_Sout.Length; i++)
            {
                if (BN_out[i] < 0)
                {
                    x_Sout[i] = 0;
                }
            }
            return x_Sout;
        }
        /// <summary>
        /// 激活函数
        /// 数组按引用传递
        /// </summary>
        /// <param name="xi"></param>
        private float[] F(float[] xi)
        {
            float[] xout = new float[xi.Length];

            for (int i = 0; i < xi.Length; i++)
            {
                xout[i] = xi[i];
                if (xout[i] < 0)
                {
                    xout[i] = 0;
                }
            }
            return xout;
        }
        private float F(float xi)
        {
            if (xi < 0)
            {
                return 0;
            }
            return xi;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="xi_input">wi*xi的输入</param>
        public void forward(float[] xi_input)
        {
            Xin = new float[xi_input.Length];

            for (int i = 0; i < Xin.Length; i++)
            {
                Xin[i] = xi_input[i];
            }
            ////不加BN
            //Xout = F(Xin);
            BN_out = BN.GetX_out(Xin);
            Xout = F(BN_out);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sensitive_input">wi*sensitive的输入</param>
        public void backpropagation(float[] cancha)
        {
            ////不加BN
            //SensitiveValue = F_back(cancha);
            float[] f_out = F_back(cancha);
            SensitiveValue = BN.GetDout(f_out, Xin);
        }
    }
}
