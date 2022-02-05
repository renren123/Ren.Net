using Ren.Device;
using Ren.Net.Loss;
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
        /// <summary>
        /// 是为了设置二维数组最大值，使Tensor 后面不用重复申请显存
        /// </summary>
        private int MaxLinearNumber { set; get; }
        private List<NetModule> Nets { set; get; }
        public Optimizer Optimizer { set; get; }
        public NetLoss Loss { set; get; }

        public DeviceTpye Device
        { 
            set 
            {
                Tensor.Device = value;
            }
            get
            {
                return Tensor.Device;
            } 
        }

        public Sequential(List<NetModule> nets)
        {
            Nets = new List<NetModule>(nets);

            for (int i = 0; i < Nets.Count; i++)
            {
                if (Nets[i] is Networks.Linear net)
                {
                    MaxLinearNumber = Math.Max(MaxLinearNumber, Math.Max(net.OutputNumber, net.InputNumber));
                }
            }
            MaxLinearNumber += 2;
            Tensor.MaxLinearNumber = this.MaxLinearNumber;
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
            Log.Debug("");
            Log.Debug("");
            Log.Debug(" ********************* net initing *********************");

            this.Optimizer.Device = this.Device;

            for (int i = 0; i < Nets.Count; i++)
            {
                var net = Nets[i];
                net.Optimizer = this.Optimizer.Clone() as Optimizer;
                net.Optimizer.MaxLinearNumber = MaxLinearNumber;
                // net 的GetWI 找下一个 GetWI 赋值的激活函数
                if (net.WIInitialize == null)
                {
                    net.WIInitialize = GetWeightsOptimizerDelegate(i + 1);
                    if (net.WIInitialize == null)   // 如果没找到 就从头找 第一个
                    {
                        net.WIInitialize = GetWeightsOptimizerDelegate(0);
                    }
                    if (net.WIInitialize == null)   // 如果没有找到就用默认的
                    {
                        net.WIInitialize = new WIInitialization();
                    }
                }

                net.Device = this.Device;
                if (Device == DeviceTpye.CUDA)
                {
                    net.MaxLinearNumber = MaxLinearNumber;
                }
                net.Init();
            }

            if (Device == DeviceTpye.CUDA)
            {
                Tensor.SwapA = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
                Tensor.SwapB = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
                Tensor.SwapC = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            }

            Log.Debug("\r\n\r\nnet: \r\n" + this.ToString());
            IsInit = true;
            Log.Debug(" ********************* net inited *********************");
            Log.Debug("");
            Log.Debug("");
        }
        /// <summary>
        /// 向下 找激活函数，然后分配 激活函数中的权限初始化 对象给 网络节点, 从 index 下一个节点
        /// 开始遍历
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        private WIInitialization GetWeightsOptimizerDelegate(int index)
        {
            for (int i = index; i < Nets.Count; i++)
            {
                var net = Nets[i];
                if(net.WIInitialize != null)
                {
                    return net.WIInitialize;
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
            // 损失函数的反向传播
            @out = Loss.Backup(@out);

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
        public static void Save(Sequential sequential, string fileName = "file.name")
        {
            if (fileName == null)
            {
                fileName =Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), fileName);
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
        public static Sequential Load(string fileName = "file.name")
        {
            if(fileName == null)
            {
                fileName = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), fileName);
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
