using Ren.Net.Optimizers;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;
using static Ren.Net.Objects.NetModule;

namespace Ren.Net.Objects
{
    public class Sequential
    {
        private bool IsInit { set; get; } = false;
        private List<NetModule> Nets { set; get; }
        public Optimizer Optimizer { set; get; }
        public Sequential(List<NetModule> nets)
        {
            Nets = new List<NetModule>(nets);
        }
        private void Init()
        {
            Log.Debug("net initing");

            if (IsInit)
            {
                return;
            }
            for (int i = 0; i < Nets.Count; i++)
            {
                var net = Nets[i];
                net.Optimizer = this.Optimizer.Clone() as Optimizer;
                if (net.GetWI == null)
                {
                    net.GetWI = GetNextWeightsDelegate(i);
                    if(net.GetWI == null && i != 0)
                    {
                        net.GetWI = GetNextWeightsDelegate(0);
                    }
                }
                net.Init();
            }

            IsInit = true;
            Log.Debug("net inited");
        }
        private WeightsDelegate GetNextWeightsDelegate(int index)
        {
            for (int i = index + 1; i < Nets.Count; i++)
            {
                var net = Nets[i];
                if(net.GetWI != null)
                {
                    return net.GetWI;
                }
            }
            return null;
        }
        public Torch Forward(Torch @in)
        {
            Init();
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
        public void OptimizerStep()
        {
            for (int i = 0; i < Nets.Count; i++)
            {
                Nets[i].Optimizer.Step();
            }
        }
        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < Nets.Count; i++)
            {
                var net = Nets[i];

                builder.AppendLine($"{i}、{net}");
            }
            return builder.ToString();
        }
    }
}
