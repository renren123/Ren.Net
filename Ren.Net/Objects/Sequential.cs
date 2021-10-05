using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    public class Sequential
    {
        private List<NetModule> Nets { set; get; }

        public Sequential(List<NetModule> nets)
        {
            Nets = new List<NetModule>(nets);
        }
        public Torch Forward(Torch @in)
        {
            for (int i = 0; i < Nets.Count; i++)
            {
                var net = Nets[i];
                @in = net.Forward(@in);
            }
            return @in;
        }
        public Torch Backup(Torch @out)
        {
            for (int i = Nets.Count - 1; i >= 0; i--)
            {
                var net = Nets[i];
                @out = net.Backup(@out);
            }
            return @out;
        }
    }
}
