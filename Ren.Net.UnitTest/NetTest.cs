using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using Ren.Net.ActivationFunction;
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
            //adamMock.Setup(p => p.GetOptimizer(It.IsAny<float>(), It.IsAny<int>(), It.IsAny<int>())).Returns((float dw, int a, int b)=> dw);
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
    }
}
