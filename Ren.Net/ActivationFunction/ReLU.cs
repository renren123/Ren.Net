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
            switch (@in.Dimension)
            {
                case 1: 
                    {
                        for (int i = 0; i < @in.Data1d.Length; i++)
                        {
                            if (@in.Data1d[i] < 0)
                            {
                                @in.Data1d[i] = 0;
                            }
                        }
                    }
                    break;
                default:
                    throw new Exception("ReLU::Forward default");
            }
            return @in;
        }
        public override Torch Backup(Torch @out)
        {
            switch (@out.Dimension)
            {
                case 1:
                    {
                        for (int i = 0; i < @out.Data1d.Length; i++)
                        {
                            if (@out.Data1d[i] < 0)
                            {
                                @out.Data1d[i] = 0;
                            }
                        }
                    }
                    break;
                default:
                    throw new Exception("ReLU::Backup default");
            }
            return @out;
        }
    }
}
