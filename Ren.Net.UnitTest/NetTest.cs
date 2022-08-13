using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using Ren.Device;
using Ren.Net.ActivationFunction;
using Ren.Net.Extensions;
using Ren.Net.Loss;
using Ren.Net.Networks;
using Ren.Net.Objects;
using Ren.Net.Optimizers;
using System.Collections.Generic;

namespace Ren.Net.UnitTest
{
    [TestClass]
    public class NetTest
    {
        [TestMethod]
        public void TestUnitFramework()
        {
            Assert.AreEqual(1 + 0.001, 1, 0.001, "Account not debited correctly");
        }
        [TestMethod]
        public void TorchTest()
        {
            //Torch input = new Torch(new List<float[]>()
            //{
            //    new float[]{1, -1}
            //});
            //var copy = input.Clone() as Torch;

            //Assert.AreEqual(1, copy.Data[0][0]);
            //Assert.AreEqual(-1, copy.Data[0][1]);
        }
        [TestMethod]
        public void ReLUTest()
        {
            //ReLU reLU = new ReLU();
            //Torch input = new Torch(new List<float[]>() 
            //{
            //    new float[]{1, -1}
            //});
            //Torch sensitive = new Torch(new List<float[]>()
            //{
            //    new float[]{2, -2}
            //});
            //var forwardOutput = reLU.Forward(input);
            //Assert.AreEqual(1, forwardOutput.Data[0][0]);
            //Assert.AreEqual(0, forwardOutput.Data[0][1]);

            //var backup = reLU.Backup(sensitive);
            //Assert.AreEqual(2, backup.Data[0][0]);
            //Assert.AreEqual(0, backup.Data[0][1]);
        }
        [TestMethod]
        public void MSELossTest()
        {
            //// https://blog.csdn.net/hao5335156/article/details/81029791
            //MSELoss loss = new MSELoss();
            //Torch label = new Torch(new List<float[]>()
            //{
            //    new float[]{1, 2},
            //    new float[]{3, 4}
            //});
            //Torch output = new Torch(new List<float[]>()
            //{
            //    new float[]{2, 3},
            //    new float[]{4, 5}
            //});
            //var result = loss.CaculateLoss(label, output);

            //Assert.AreEqual(1, result.Data[0][0]);
            //Assert.AreEqual(1, result.Data[0][1]);
            //Assert.AreEqual(1, result.Data[1][0]);
            //Assert.AreEqual(1, result.Data[1][1]);
        }
        [TestMethod]
        public void LinearTest()
        {
            Tensor.Device = Device.DeviceTpye.CPU;
            Linear linear1 = new LinearCPU(2, 2);
            MSELoss loss = new MSELoss();

            linear1.Optimizer = new SGDCPU(0.01F);

            linear1.WI = new Tensor(new float[,]
            {
                { 1F, -1.5F, 1F },
                { 1F, 2F, -1F },
            });
            Tensor input = new Tensor(new float[,]
            {
                { 1, 6 },
                { 2, 3 },
            });
            Tensor label = new Tensor(new float[,]
            {
                { 2, 4 },
                { 3, 5 },
            });
            Tensor one_out_label = new Tensor(new float[,]
            {
                { -1.0000F, 2.5000F},
                { 4.0000F, 11.0000F},
            });
            Tensor two_out_label = new Tensor(new float[,]
            {
                { -0.8125F, 3.0400F},
                { 3.5800F, 9.5550F},
            });

            var output = linear1.Forward(input);
            Assert.AreEqual(true, output.EqualsValue(one_out_label, 4));

            Tensor sensitive = loss.CaculateLoss(label, output);
            Tensor backUp = loss.Backward(sensitive);

            linear1.Backward(backUp);
            output = linear1.Forward(input);
            Assert.AreEqual(true, output.EqualsValue(two_out_label, 4));

            //var linearMock = new Mock<Linear>(2, 2);
            //linearMock.Setup(p => p.W_value_method(2)).Returns(1);
            ////linearMock.SetupProperty(p => p.WI , new List<float[]>()
            ////{
            ////    new float[] {1 , 2},
            ////    new float[] {3 , 4}
            ////});
            //linearMock.Object.WI = new List<float[]>()
            //{
            //    new float[] {1 , 2},
            //    new float[] {3 , 4}
            //};
            //linearMock.CallBase = true;

            //var adamMock = new Mock<Adam>(1);
            //adamMock.Setup(p => p.GetOptimizer(It.IsAny<float>(), It.IsAny<int>(), It.IsAny<int>())).Returns((float dw, int a, int b) => dw);
            //adamMock.CallBase = true;

            //Linear linear = linearMock.Object;
            //linear.Optimizer = adamMock.Object;

            //Torch input = new Torch(new List<float[]>()
            //{
            //    new float[]{2},
            //    new float[]{3}
            //});
            //var output = linear.Forward(input);

            //Assert.AreEqual(true, output != null);
            //Assert.AreEqual(8, output.Data[0][0]);
            //Assert.AreEqual(18, output.Data[1][0]);

            //Torch sensitive = new Torch(new List<float[]>()
            //{
            //    new float[]{1},
            //    new float[]{1}
            //});
            //var backupOutput = linear.Backup(sensitive);

            //Assert.AreEqual(4, backupOutput.Data[0][0]);
            //Assert.AreEqual(6, backupOutput.Data[1][0]);

            //Assert.AreEqual(-1, linear.WI[0][0]);
            //Assert.AreEqual(-1, linear.WI[0][1]);
            //Assert.AreEqual(1, linear.WI[1][0]);
            //Assert.AreEqual(1, linear.WI[1][1]);
        }
        [TestMethod]
        public void BatchNorm1DCPUTest()
        {
            // BN 前向传播 和 反向传播
            // https://blog.csdn.net/weixin_39228381/article/details/107896863
            Tensor.Device = Device.DeviceTpye.CPU;
            BatchNorm1DCPU batchNorm1D = new BatchNorm1DCPU(4);
            MSELoss loss = new MSELoss();
            batchNorm1D.Optimizer = new SGDCPU(1F);

            batchNorm1D.Init();
            Tensor input = new Tensor(new float[,]
            {
                { 1, 6, 2},
                { 2, 3, 4},
                { 4, 2, 6},
                { 1, 4, 1},
            });
            Tensor label = new Tensor(new float[,]
            {
                { 1, 2, 1},
                { 2, 3, 2},
                { 3, 4, 1},
                { 4, 5, 2},
            });
            Tensor one_out_label = new Tensor(new float[,]
            {
                { -0.9258F, 1.3887F, -0.4629F},
                { -1.2247F, 0.0000F, 1.2247F},
                { 0.0000F, -1.2247F, 1.2247F},
                { -0.7071F, 1.4142F, -0.7071F},
            });

            Tensor tow_out_label = new Tensor(new float[,]
            {
                { -0.0105F, 1.6825F, 0.3281F},
                { 0.5543F, 1.1667F, 1.7790F},
                { 1.3333F, 1.4710F, 1.1957F},
                { 1.1464F, 3.2071F, 1.1464F},
            });

            Tensor three_out_label = new Tensor(new float[,]
            {
                { 0.4471F, 1.8293F, 0.7236F},
                { 1.4438F, 1.7500F, 2.0562F},
                { 2.0000F, 2.8188F, 1.1812F},
                { 2.0732F, 4.1036F, 2.0732F},
            });
            var output = batchNorm1D.Forward(input);
            Assert.AreEqual(true, output.EqualsValue(one_out_label, 4));

            Tensor sensitive = loss.CaculateLoss(label, output);
            Tensor backUp = loss.Backward(sensitive);
            batchNorm1D.Backward(backUp);
            output = batchNorm1D.Forward(input);
            Assert.AreEqual(true, output.EqualsValue(tow_out_label, 4));

            sensitive = loss.CaculateLoss(label, output);
            backUp = loss.Backward(sensitive);
            batchNorm1D.Backward(backUp);
            output = batchNorm1D.Forward(input);
            Assert.AreEqual(true, output.EqualsValue(three_out_label, 4));
        }
        [TestMethod]
        public void BatchNorm1DCPUForwardTest()
        {
            // 每一行是一个batch, size = 2
            Tensor.Device = Device.DeviceTpye.CPU;
            BatchNorm1DCPU batchNorm1D = new BatchNorm1DCPU(10);
            batchNorm1D.Optimizer = new SGDCPU(1F);
            batchNorm1D.Init();
            Tensor input = new Tensor(new float[,]
            {
                { 0, 10 },
                { 1, 11 },
                { 2, 12 },
                { 3, 13 },
                { 4, 14 },
                { 5, 15 },
                { 6, 16 },
                { 7, 17 },
                { 8, 18 },
                { 9, 19 },
            });
            Tensor label = new Tensor(new float[,]
            {
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
                { -1.0000F, 1.0000F },
            });
            Tensor output = batchNorm1D.Forward(input);

            Assert.AreEqual(true, output.EqualsValue(label, 4));
        }
        [TestMethod]
        public void SoftMaxTest()
        {
            SoftMax softMax = new SoftMax();
            Tensor.Device = Device.DeviceTpye.CPU;

            softMax.Init();
            Tensor input = new Tensor(new float[,]
            {
                { 1, 2, 3 },
                { -1, -2, -3}
            });
            Tensor label = new Tensor(new float[,]
            {
                { 0.09003057F, 0.24472847F, 0.66524096F },
                { 0.66524096F, 0.24472847F, 0.09003057F}
            });
            Tensor sensitive = new Tensor(new float[,]
            {
                { 1, 2, -1 },
                { -1, 1, 0.5F}
            });
            Tensor sensitiveLabel = new Tensor(new float[,]
            {
                { 0.30834F, -0.09004F, -1.17295F },
                { -3.04362F, -3.64276F, -3.86317F}
            });
            // forward
            Tensor output = softMax.Forward(input);
            Assert.AreEqual(true, output.EqualsValue(label, 6));
            Tensor outputSum = output.Sum(axis: AxisType.Row);
            for (int i = 0; i < outputSum.Column; i++)
            {
                Assert.AreEqual(1, outputSum[0, i]);
            }
            // backward
            Tensor sensitiveOutput = softMax.Backward(sensitive);
            Assert.AreEqual(true, sensitiveOutput.EqualsValue(sensitiveLabel, 6));
        }
    }
}
