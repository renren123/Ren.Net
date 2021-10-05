using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.ActivationFunction
{
    public class ReLU : NetModule
    {
        public override Torch Forward(Torch @in)
        {
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
            foreach (var item in @out.Data)
            {
                for (int i = 0; i < item.Length; i++)
                {
                    if (item[i] < 0)
                    {
                        item[i] = 0;
                    }
                }
            }
            return @out;
        }
    }
}
