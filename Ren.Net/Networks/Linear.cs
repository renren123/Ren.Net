using Ren.Device;
using Ren.Net.Objects;
using Ren.Net.Optimizers;
using Ren.Net.Util;
using Serilog;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Ren.Net.Networks
{
    [Serializable]
    public class Linear : NetModule
    {
        public static Tensor SwapA { set; get; }
        public static Tensor SwapB { set; get; }


        /// <summary>
        /// 输入层神经元个数
        /// </summary>
        public int InputNumber { set; get; }
        /// <summary>
        /// 输出层神经元个数
        /// </summary>
        public int OutputNumber { set; get; }
        /// <summary>
        /// 权重数组
        /// </summary>
        public Tensor WI { set; get; }
        /// <summary>
        /// list 的数量是前一层的数量
        /// </summary>
        public Tensor X_In { set; get; }

        public Linear LinearDevice { set; get; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputSize">输入层 神经元数量</param>
        /// <param name="outputSize">当前层 神经元的数量</param>
        public Linear(int inputNumber, int outputNumber)
        {
            this.InputNumber = inputNumber;
            this.OutputNumber = outputNumber;
        }
        public override void Init()
        {
            this.LinearDevice = InstenceHelper<Linear>.GetInstence(
                typeof(Linear), 
                new object[] { this.InputNumber, this.OutputNumber }).
                Find(p => p.Device == this.Device);
            //var type = typeof(Linear).Assembly.GetTypes().Where(type => type.IsSubclassOf(typeof(Linear)));
            //List<Linear> linears = new List<Linear>();
            //foreach (var item in type)
            //{
            //    linears.Add(Activator.CreateInstance(item, new object[] { this.InputNumber, this.OutputNumber }) as Linear);
            //}
            //this.LinearDevice = linears.Find(p => p.Device == this.Device);
            Optimizer.InputNumber = this.InputNumber + 1;
            Optimizer.OutputNumber = this.OutputNumber;
            Optimizer.Init();
            this.LinearDevice.Optimizer = Optimizer;
            this.LinearDevice.WIOptimizer = WIOptimizer;
            this.LinearDevice.MaxLinearNumber = this.MaxLinearNumber;


            this.LinearDevice.Init();

            return;


            Optimizer.InputNumber = this.InputNumber + 1;
            Optimizer.OutputNumber = this.OutputNumber;
            Optimizer.Init();

            int sumInput = OutputNumber + InputNumber;
            //WI = new Tensor(OutputNumber, InputNumber + 1, (int i, int j) =>
            //{
            //    if (j == InputNumber)
            //    {
            //        return 1F;
            //    }
            //    else
            //    {
            //        return WIOptimizer.GetWI(sumInput);
            //    }
            //});

            WI = new Tensor(MaxLinearNumber, MaxLinearNumber, (int i, int j) =>
            {
                if (j == InputNumber)
                {
                    return 1F;
                }
                else
                {
                    return WIOptimizer.GetWI(sumInput);
                }
            });
            WI.Width = OutputNumber;
            WI.Height = InputNumber + 1;

            //SwapA = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            //SwapB = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            X_In = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);

            Log.Debug($"Linear inited [{InputNumber}, {OutputNumber}]");
        }
        public override Tensor Forward(Tensor @in)
        {
            int batchSize = @in.Height == 0 ? @in.Column : @in.Height;          // batch 的大小

            if (batchSize <= 0)
            {
                throw new Exception($"Linear::Forward, batchSize {batchSize}");
            }
            return this.LinearDevice.Forward(@in);



            switch (Tensor.Device)
            {
                case DeviceTpye.CPU:
                    {
                        X_In = @in.AddOneRowWithValue(batchSize, 1F);
                        @in = WI * X_In;
                    }
                    break;
                case DeviceTpye.CUDA:
                    {
                        Tensor.AddLastOneRowWithValue(@in, 1F, X_In);
                        Tensor.Multiply(WI, X_In, @in);
                    }
                    break;
                default:
                    throw new Exception($"Linear::Forward, Device {Tensor.Device}");
            }


            //X_In = @in.Clone() as Tensor;    // 保存输入，用于反向传播时更新 WI 的大小
            //@in = @in.AddOneRowWithValue(batchSize, 1F);
            //Tensor x_out = WI * @in;


            // ********************** Test **********************
            //if (X_In != null)
            //{
            //    X_In.Dispose();
            //}
            // X_In = @in.AddOneRowWithValue(batchSize, 1F);    // 保存输入，用于反向传播时更新 WI 的大小
            // Tensor x_out = WI * X_In;
            // ********************** Test *********************

            // ********************** Test *********************
            //Tensor.AddLastOneRowWithValue(@in, 1F, X_In);
            //Tensor.Multiply(WI, X_In, @in);
            //Tensor.RemoveLastOneRow(X_In);
            // ********************** Test *********************


            // @in.Dispose();
            return @in;
        }
        /// <summary>
        /// 反向传播
        /// </summary>
        /// <param name="out">list 是当前层神经元的数量</param>
        /// <returns></returns>
        public override Tensor Backup(Tensor @out)    // wi 数量是上一层神经元的数量，假设out 里面 是 误差值
        {
            // batchSize
            if (@out.Column <= 0)
            {
                throw new Exception($"Linear::Backup, batchSize is {@out.Column}");
            }
            return this.LinearDevice.Backup(@out);

            switch (Tensor.Device)
            {
                case DeviceTpye.CPU:
                    {
                        using Tensor sensitive_out = WI.Transpose() * @out;
                        using Tensor dwTemp = @out * X_In.Transpose();
                        WI -= Optimizer.GetOptimizer(dwTemp, null);
                        return sensitive_out.RemoveLastOneRow();
                    }
                    break;
                case DeviceTpye.CUDA:
                    {
                        Tensor.Transpose(WI);
                        Tensor.Copy(@out, Tensor.SwapA);
                        Tensor.Multiply(WI, Tensor.SwapA, SwapA);    // SwapA = sensitive_out
                        Tensor.Transpose(WI);
                        Tensor.Transpose(X_In);
                        Tensor.Multiply(@out, X_In, Tensor.SwapA);          // dwTemp = Tensor.Temp1

                        Tensor.Copy(SwapA, @out);
                        Tensor.Copy(Tensor.SwapA, SwapA);
                        Tensor.RemoveLastOneRow(@out);

                        Optimizer.GetOptimizer(SwapA, SwapB);

                        Tensor.Minus(WI, SwapB, WI);
                    }
                    break;
                default:
                    throw new Exception($"Linear::Backup, Device {Tensor.Device}");
            }


            //Tensor.Transpose(WI);
            //Tensor.Copy(@out, Tensor.SwapA);
            //Tensor.Multiply(WI, Tensor.SwapA, SwapA);    // SwapA = sensitive_out
            //Tensor.Transpose(WI);
            //Tensor.Transpose(X_In);
            //Tensor.Multiply(@out, X_In, Tensor.SwapA);          // dwTemp = Tensor.Temp1

            //Tensor.Copy(SwapA, @out);
            //Tensor.Copy(Tensor.SwapA, SwapA);
            //Tensor.RemoveLastOneRow(@out);

            //Optimizer.GetOptimizer(SwapA, SwapB);

            //Tensor.Minus(WI, SwapB, WI);
            return @out;


            //using Tensor wiT = WI.Transpose();

            //using Tensor sensitive_out = wiT * @out;

            //using Tensor xinT = X_In.Transpose();

            //using Tensor dwTemp = @out * xinT; 

            ////using Tensor delta = Optimizer.GetOptimizer(dwTemp, Swap);

            ////WI.MinusToA(delta);

            //// WI -= Optimizer.GetOptimizer(dwTemp);

            //return sensitive_out.RemoveLastOneRow();
        }
        public override string ToString()
        {
            return $"Linear [{InputNumber}, {OutputNumber}]";
        }
    }
}
