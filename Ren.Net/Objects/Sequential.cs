using Ren.Device;
using Ren.Net.Optimizers;
using Serilog;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using static Ren.Net.Objects.NetModule;

namespace Ren.Net.Objects
{
    [Serializable]
    public class Sequential
    {
        private bool IsInit { set; get; } = false;
        private List<NetModule> Nets { set; get; }
        public Optimizer Optimizer { set; get; }
        public DeviceTpye Device { set; get; } = DeviceTpye.CUDA;

        public Sequential(List<NetModule> nets)
        {
            Nets = new List<NetModule>(nets);
        }
        /// <summary>
        /// 初始化
        /// </summary>
        private void Init()
        {
            if (IsInit)
            {
                return;
            }
            Log.Debug(" ********************* net initing *********************");

            switch (Device)
            {
                case DeviceTpye.CPU:
                    {

                        for (int i = 0; i < Nets.Count; i++)
                        {
                            var net = Nets[i];
                            net.Optimizer = this.Optimizer.Clone() as Optimizer;
                            // net 的GetWI 找下一个 GetWI 赋值的激活函数
                            if (net.WIOptimizer == null)
                            {
                                net.WIOptimizer = GetNextWeightsDelegate(i);
                                if (net.WIOptimizer == null && i != 0) // 如果没找到 就从头找 第一个
                                {
                                    net.WIOptimizer = GetNextWeightsDelegate(0);
                                }
                            }
                            net.Init();
                        }
                    }
                    break;
                case DeviceTpye.CUDA:
                    {
                        int maxLinearNumber = 0;
                        for (int i = 0; i < Nets.Count; i++)
                        {
                            if (Nets[i] is Networks.Linear net)
                            {
                                maxLinearNumber = Math.Max(maxLinearNumber, Math.Max(net.OutputNumber, net.InputNumber));
                            }
                        }
                        maxLinearNumber += 2;

                        for (int i = 0; i < Nets.Count; i++)
                        {
                            var net = Nets[i];
                            net.Optimizer = this.Optimizer.Clone() as Optimizer;
                            net.Optimizer.MaxLinearNumber = maxLinearNumber;
                            // net 的GetWI 找下一个 GetWI 赋值的激活函数
                            if (net.WIOptimizer == null)
                            {
                                net.WIOptimizer = GetNextWeightsDelegate(i);
                                if (net.WIOptimizer == null && i != 0) // 如果没找到 就从头找 第一个
                                {
                                    net.WIOptimizer = GetNextWeightsDelegate(0);
                                }
                            }
                            net.MaxLinearNumber = maxLinearNumber;
                            net.Init();
                        }

                        Tensor.SwapA = new Tensor(maxLinearNumber, maxLinearNumber, 0F);
                        Tensor.SwapB = new Tensor(maxLinearNumber, maxLinearNumber, 0F);
                        Tensor.SwapC = new Tensor(maxLinearNumber, maxLinearNumber, 0F);

                        Networks.Linear.SwapA = new Tensor(maxLinearNumber, maxLinearNumber, 0F);
                        Networks.Linear.SwapB = new Tensor(maxLinearNumber, maxLinearNumber, 0F);
                    }
                    break;
                default:
                    throw new Exception("Sequential::Init");
            }

            Log.Debug("\r\n\r\nnet: \r\n" + this.ToString());
            IsInit = true;
            Log.Debug(" ********************* net inited *********************");
        }
        /// <summary>
        /// 向下 找激活函数，然后分配 激活函数中的权限初始化 对象给 网络节点
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        private WIOptimizer GetNextWeightsDelegate(int index)
        {
            for (int i = index + 1; i < Nets.Count; i++)
            {
                var net = Nets[i];
                if(net.WIOptimizer != null)
                {
                    return net.WIOptimizer;
                }
            }
            return null;
        }
        /// <summary>
        /// 输入 行是神经元的个数，列是 batchsize
        /// </summary>
        /// <param name="in"></param>
        /// <returns></returns>
        public Tensor Forward(Tensor @in)
        {
            Init();
            for (int i = 0; i < Nets.Count; i++)
            {
                var net = Nets[i];
                @in = net.Forward(@in);
            }
            return @in;
        }
        public Tensor Backup(Tensor @out)
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
        public static void Save(Sequential sequential, string fileName = null)
        {
            if (fileName == null)
            {
                fileName =Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), "file.name");
            }
            FileStream fs = new FileStream(fileName, FileMode.Create);
            try
            {
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(fs, sequential);
            }
            catch(Exception ew)
            {
                Log.Error(ew.ToString());
            }
            finally
            {
                fs.Close();
            }
        }
        public static Sequential Load(string fileName = null)
        {
            if(fileName == null)
            {
                fileName = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), "file.name");
            }
            Sequential sequential = null;
            
            FileStream fs = new FileStream(fileName, FileMode.Open);
            try
            {
                BinaryFormatter formatter = new BinaryFormatter();
                sequential = formatter.Deserialize(fs) as Sequential;
            }
            catch (Exception ew)
            {
                Log.Error(ew.ToString());
            }
            finally
            {
                fs.Close();
            }
            return sequential;
        }
    }
}
