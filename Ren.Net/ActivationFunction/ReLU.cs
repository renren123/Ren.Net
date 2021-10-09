using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.ActivationFunction
{
    public class ReLU : NetModule
    {
        private Torch X_IN;
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
    }
}
