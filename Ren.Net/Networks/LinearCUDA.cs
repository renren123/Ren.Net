using Ren.Device;
using Ren.Net.Objects;
using Serilog;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Networks
{
    [Serializable]
    public class LinearCUDA : Linear
    {
        public override DeviceTpye Device { get => DeviceTpye.CUDA; }
        private Tensor SwapA { set; get; }
        private Tensor SwapB { set; get; }
        private Tensor SwapC { set; get; }

        private Dictionary<Tensor, int> SwapDic { set; get; } = new Dictionary<Tensor, int>();

        public LinearCUDA(int inputNumber, int outputNumber) : base(inputNumber, outputNumber)
        {
            
        }
        public override void Init()
        {
            int sumInput = OutputNumber + InputNumber;
            WI = new Tensor(MaxLinearNumber, MaxLinearNumber, (int i, int j) =>
            {
                if (j == InputNumber)
                {
                    return 1F;
                }
                else
                {
                    return WIInitialize.GetWI(sumInput);
                }
            });
            WI.Width = OutputNumber;
            WI.Height = InputNumber + 1;

            X_In = new Tensor(MaxLinearNumber, MaxLinearNumber, 0F);
            Log.Debug($"Linear CUDA inited [{InputNumber}, {OutputNumber}]");
        }
        public override Tensor Forward(Tensor @in)
        {
            Tensor.AddLastOneRowWithValue(@in, 1F, X_In);
            Tensor.Multiply(WI, X_In, @in);
            return @in;
        }
        public override Tensor Backup(Tensor @out)
        {
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
            //return @out;

            LoadPublicValue();
            Tensor.Transpose(WI);
            Tensor.Copy(@out, SwapC);
            Tensor.Multiply(WI, SwapC, SwapA);    // SwapA = sensitive_out
            Tensor.Transpose(WI);
            Tensor.Transpose(X_In);
            Tensor.Multiply(@out, X_In, SwapC);          // dwTemp = Tensor.Temp1

            Tensor.Copy(SwapA, @out);
            Tensor.Copy(SwapC, SwapA);
            Tensor.RemoveLastOneRow(@out);

            SetPublicValue(SwapC);
            Optimizer.GetOptimizer(SwapA, SwapB);

            Tensor.Minus(WI, SwapB, WI);

            SetPublicValue(SwapA, SwapB, SwapC);
            return @out;
        }
        public void LoadPublicValue()
        {
            SwapA = NetParameter.GetRegisterParameter();
            SwapB = NetParameter.GetRegisterParameter();
            SwapC = NetParameter.GetRegisterParameter();
            if (SwapA == null || SwapB == null || SwapC == null)
            {
                throw new Exception("can not find enough swap tensor");
            }
        }
    }
}
