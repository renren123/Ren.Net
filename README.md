# Ren.Net
模仿 pytorch 使用 .net 构造神经网络系统

mathnet 分支中，网络使用 mathnet-numerics 科学计算库使能矩阵计算

master 分支则是正常使用 遍历的方式 实现前向传播和反向传播

# mathnet-numerics 地址：
https://github.com/mathnet/mathnet-numerics

# CUDA 加速
https://github.com/m4rs-mt/ILGPU

# ADAM 
为例来进行显存控制，将整个过程进行分解，然后利用中间变量消除重复申请显存的问题
![image](https://user-images.githubusercontent.com/26969703/140619034-0bb65d69-1112-4a65-90ad-cafbc76fd8a2.png)

