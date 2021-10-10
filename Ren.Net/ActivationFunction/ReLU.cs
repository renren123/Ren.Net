using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.ActivationFunction
{
    public class ReLU : NetModule
    {
        private readonly Random r = new Random(DateTime.UtcNow.Millisecond);
        private Torch X_IN;

        public ReLU()
        {
            this.GetWI = W_value_method;
        }

        public override void Init()
        {
        }
        public override Torch Forward(Torch @in)
        {
            X_IN = @in.Clone() as Torch;

            for (int i = 0; i < @in.Row; i++)
            {
                for (int j = 0; j < @in.Column; j++)
                {
                    if(@in[i, j] < 0)
                    {
                        @in[i, j] = 0;
                    }
                }
            }
            return @in;
        }
        public override Torch Backup(Torch @out)
        {
            for (int i = 0; i < X_IN.Row; i++)
            {
                for (int j = 0; j < X_IN.Column; j++)
                {
                    if (X_IN[i, j] < 0)
                    {
                        @out[i, j] = 0;
                    }
                }
            }
            return @out;
        }
        private float W_value_method(int sumInput)
        {
            // return 1F;

            //float x = (float)r.NextDouble();
            //float number = (Math.Abs(x) / 1) * (2.0F / sumInput);

            float y = (float)r.NextDouble();
            float x = (float)r.NextDouble();
            float number = (float)(Math.Cos(2 * Math.PI * x) * Math.Sqrt(-2 * Math.Log(1 - y)));
            number *= (float)Math.Sqrt(2.0 / sumInput);
            number = Math.Abs(number);
            number *= (2.0F / sumInput);

            //float y = (float)r.NextDouble();
            //float x = (float)r.NextDouble();
            //float number = (float)(Math.Cos(2 * Math.PI * x) * Math.Sqrt(-2 * Math.Log(1 - y)));
            //number *= (float)Math.Sqrt(2.0 / sumInput);
            //number = Math.Abs(number);

            return number;
        }
        public override string ToString()
        {
            return "ReLU";
        }
    }
}
