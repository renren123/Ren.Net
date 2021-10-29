using System;
using System.Threading;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Ren.Device;

namespace Ren.Net.Test
{
    public interface NetInterface
    {
        public int Column { get; set; }
        public int Row { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public NetInterface Multy(NetInterface rhs);
        public static NetInterface operator *(NetInterface lhs, NetInterface rhs)
        {
            return lhs.Multy(rhs);
        }
    }
    public class MatrxTest : NetInterface
    {
        public int Column { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public NetInterface Multy(NetInterface rhs)
        {
            throw new NotImplementedException();
        }
    }
    public class GpuTest
    {
        static void Main(string[] args)
        {
            //NetInterface net1 = new MatrxTest();

            //NetInterface net2 = new MatrxTest();

            //var number = net1.Column;

            //var result = net1 * net2;


            ILGPUNet a = new ILGPUNet(10000, 10000, 1F);
            ILGPUNet b = new ILGPUNet(10000, 10000, 1F);

            Parallel.For(0, 1000, (i) => 
            {
                
            });


            for (int i = 0; i < 10000000; i++)
            {
                using var result = b.Add(b);
                //result.Dispose();
                // b.AddToA(a);
            }
           
            //while (true)
            //{
            //    GpuNetBase test = new GpuNetBase();


            //    test.Test();
            //    Thread.Sleep(100);
            //}
            Console.WriteLine("end");
            Console.ReadKey();
        }
    }
}
