﻿using Ren.Net.Objects;
using Ren.Net.Optimizers;
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
            Optimizer.InputNumber = this.InputNumber + 1;
            Optimizer.OutputNumber = this.OutputNumber;
            Optimizer.Init();

            int sumInput = OutputNumber + InputNumber;
            WI = new Tensor(OutputNumber, InputNumber + 1, (int i, int j) =>
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

            Log.Debug($"Linear inited [{InputNumber}, {OutputNumber}]");
        }
        public override Tensor Forward(Tensor @in)
        {
            int batchSize = @in.Column;          // batch 的大小

            if (batchSize <= 0)
            {
                throw new Exception($"Linear::Forward, batchSize {batchSize}");
            }
            //X_In = @in.Clone() as Tensor;    // 保存输入，用于反向传播时更新 WI 的大小
            //@in = @in.AddOneRowWithValue(batchSize, 1F);
            //Tensor x_out = WI * @in;


            // ********************** Test **********************
            X_In = @in.AddOneRowWithValue(batchSize, 1F);    // 保存输入，用于反向传播时更新 WI 的大小
            Tensor x_out = WI * X_In;
            // ********************** Test *********************

            @in.Dispose();
            return x_out;
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

            Tensor sensitive_out = WI.Transpose() * @out;

            //X_In = X_In.AddOneRowWithValue(batchSize, 1F);

            // ********************** Test **********************
            // ********************** Test **********************

            var dwTemp = @out * X_In.Transpose();

            @out.Dispose();

            WI -= Optimizer.GetOptimizer(dwTemp);

            Tensor sensitiveOut = sensitive_out.RemoveLastOneRow();
            sensitive_out.Dispose();
            return sensitiveOut;
        }
        public override string ToString()
        {
            return $"Linear [{InputNumber}, {OutputNumber}]";
        }
    }
}
