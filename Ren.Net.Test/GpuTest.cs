using Ren.Gpu;
using System;
using System.Threading;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ren.Net.Test
{
    class GpuTest
    {
        static void Main(string[] args)
        {
            while (true)
            {
                GpuNetBase test = new GpuNetBase();

                
                test.Test();
                Thread.Sleep(100);
            }
            Console.ReadKey();
        }
    }
}
