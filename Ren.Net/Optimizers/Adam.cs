using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    public class Adam : Optimizer
    {
        private float B1_Pow { set; get; } = 0.9F;
        private float B2_Pow { set; get; } = 0.999F;

        public float E { get; } = 0.00000001F;
        public float B1 { get; } = 0.9F;
        public float B2 { get; } = 0.999F;

        public List<float[]> V { set; get; }
        public List<float[]> S { set; get; }
        public Adam(float learningRate) : base(learningRate) { }
        public override void Init()
        {
            if (S == null)
            {
                S = new List<float[]>(OutputNumber);
                for (int i = 0; i < OutputNumber; i++)
                {
                    S.Add(new float[InputNumber]);
                }
            }
            if (V == null)
            {
                V = new List<float[]>(OutputNumber);
                for (int i = 0; i < OutputNumber; i++)
                {
                    V.Add(new float[InputNumber]);
                }
            }
        }
        /// <summary>
        /// int OutputIndex, int InputIndex 对应一条权重, 原理：https://www.jianshu.com/p/aebcaf8af76e
        /// </summary>
        /// <param name="dw"></param>
        /// <param name="OutputIndex"></param>
        /// <param name="InputIndex"></param>
        /// <returns></returns>
        public override float GetOptimizer(float dw, int OutputIndex, int InputIndex)
        {
            V[OutputIndex][InputIndex] = B1 * V[OutputIndex][InputIndex] + (1 - B1) * dw;
            S[OutputIndex][InputIndex] = B2 * S[OutputIndex][InputIndex] + (1 - B2) * dw * dw;

            float Vcorrection = V[OutputIndex][InputIndex] / (1 - B1_Pow);
            float Scorrection = S[OutputIndex][InputIndex] / (1 - B2_Pow);

            return LearningRate * Vcorrection / ((float)Math.Sqrt(Scorrection) + E);
        }


        public override void Step()
        {
            B1_Pow *= B1;
            B2_Pow *= B2;
        }
        public override object Clone()
        {
            if(OutputNumber == -1)
            {
                return new Adam(this.LearningRate);
            }
            Adam adam = new Adam(this.LearningRate)
            {
                OutputNumber = this.OutputNumber,
                InputNumber = this.InputNumber
            };
            if(this.S == null || this.V == null)
            {
                return adam;
            }
            adam.S = new List<float[]>(OutputNumber);
            adam.V = new List<float[]>(OutputNumber);
            for (int i = 0; i < OutputNumber; i++)
            {
                float[] tempS = new float[InputNumber];
                float[] tempV = new float[InputNumber];

                for (int j = 0; j < InputNumber; j++)
                {
                    tempS[j] = this.S[i][j];
                    tempV[j] = this.V[i][j];
                }
                adam.S.Add(tempS);
                adam.V.Add(tempV);
            }
            return adam;
        }
    }
}
