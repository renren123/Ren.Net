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

            foreach (var item in @in.Data)
            {
                for (int i = 0; i < item.Length; i++)
                {
                    if(item[i] < 0)
                    {
                        item[i] = 0;
                    }
                }
            }
            return @in;
        }
        public override Torch Backup(Torch @out)
        {
            for (int i = 0; i < X_IN.Data.Count; i++)
            {
                for (int j = 0; j < X_IN.Data[i].Length; j++)
                {
                    if(X_IN.Data[i][j] < 0)
                    {
                        @out.Data[i][j] = 0;
                    }
                }
            }
            return @out;
        }
    }
}
