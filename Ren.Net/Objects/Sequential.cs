using Ren.Net.Optimizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Objects
{
    public class Sequential
    {
        private List<NetModule> Nets { set; get; }
        public Optimizer Optimizer { set; get; }

        public Sequential(List<NetModule> nets)
        {
            Nets = new List<NetModule>(nets);
        }
        public Torch Forward(Torch @in)
        {
            for (int i = 0; i < Nets.Count; i++)
            {
                var net = Nets[i];

                if (net.Optimizer == null)
                {
                    net.Optimizer = this.Optimizer.Clone() as Optimizer;
                }
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
                net.Optimizer.Step();
            }

            return @out;
        }
        public void OptimizerStep()
        {
            for (int i = 0; i < Nets.Count; i++)
            {
                Nets[i].Optimizer.Step();
            }
        }
    }
}
