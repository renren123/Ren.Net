using Microsoft.VisualStudio.TestTools.UnitTesting;

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
    }
}
