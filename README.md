# Ren.Net
模仿 pytorch 使用 .net 构造神经网络系统

mathnet 分支中，网络使用 mathnet-numerics 科学计算库使能矩阵计算

master 分支则是正常使用 遍历的方式 实现前向传播和反向传播

# mathnet-numerics 地址：
https://github.com/mathnet/mathnet-numerics

# CUDA 加速
https://github.com/m4rs-mt/ILGPU

# ADAM 显存优化
以 Adam 优化器为例来进行显存控制，将整个过程进行分解，然后利用中间变量消除重复申请显存的问题
![image](https://user-images.githubusercontent.com/26969703/140619034-0bb65d69-1112-4a65-90ad-cafbc76fd8a2.png)

'''

        public override Tensor GetOptimizer(Tensor dw, Tensor @out)
        {
            Tensor.DotMultiply(dw, dw, Tensor.SwapA);
            Tensor.Multiply((1 - B2), Tensor.SwapA, Tensor.SwapB);
            Tensor.Multiply(B2, STorch, Tensor.SwapA);
            Tensor.Add(Tensor.SwapA, Tensor.SwapB, STorch);     // 计算出 STorch
            Tensor.Multiply(B1, VTorch, Tensor.SwapA);
            Tensor.Multiply((1 - B1), dw, Tensor.SwapB);
            Tensor.Add(Tensor.SwapA, Tensor.SwapB, VTorch);     // 计算出 VTorch
            Tensor.DotDivide(VTorch, (1 - B1_Pow), Tensor.SwapA);
            Tensor.Multiply(LearningRate, Tensor.SwapA, Tensor.SwapB);  // 计算出 Tensor.SwapB = dividend
            Tensor.DotDivide(STorch, (1 - B2_Pow), Tensor.SwapA);
            Tensor.Sqrt(Tensor.SwapA);
            Tensor.Add(Tensor.SwapA, E, Tensor.SwapC);          // 计算出 divisor， Tensor.SwapC = divisor
            Tensor.DotDivide(Tensor.SwapB, Tensor.SwapC, @out);
            return @out;
        }
        
'''

Tensor.SwapA、Tensor.SwapB、Tensor.SwapC 是三个全局变量类似寄存器的作用，这样可以不用反复申请显存
