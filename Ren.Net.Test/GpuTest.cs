using System;
using System.Threading;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Ren.Device;
using System.Diagnostics;
using ILGPU.Runtime;
using Ren.Net.Objects;

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

            Tensor a = new Tensor(new float[,] 
            {
                {1, -1, 4},
                {1, 2, 3}
            });
            a.Width = 2;
            a.Height = 2;
            Tensor b = new Tensor(new float[,]
            {
                {1, 2, 3},
                {1, 5, 3}
            });
            b.Width = 2;
            b.Height = 2;
            Tensor c = new Tensor(new float[,]
            {
                {1, 2, 3},
                {1, 2, 3},
                {1, 2, 3}
            });
            c.Width = 2;
            c.Height = 1;

            // Tensor.Multiply(a, b, c);
            // Tensor.Multiply(3F, a,  c);
            // Tensor.Add(a, b, c);
            // Tensor.Add(a, 3F, c);
            // Tensor.Minus(a, b, c);
            // Tensor.DotMultiply(a, b, c);
            // Tensor.DotDivide(a, b, c);
            // Tensor.DotDivide(a, 2F, c);
            // Tensor.Sqrt(a);

            //Tensor.AddLastOneRowWithValue(a, 9F, c);
            //PrintArray(c.ToArray());
            //Tensor.Transpose(c);

            //Tensor.DotDivide(a, 2.0F, c);


            //PrintArray(c.ToArray());
            //Tensor.Transpose(c);

            //Tensor.DotDivide(a, 2.0F, c);

            // Tensor.Copy(a, b);

            Tensor.Relu(a, b, c);


            PrintArray(c.ToArray());

            //ILGPUNet.Add(a, b, c);

            Tensor.AddLastOneRowWithValue(b, 100F, c);


            Tensor.Copy(a, b);
            PrintArray(b.ToArray());
            Console.WriteLine();
            Tensor.RemoveLastOneRow(b);
            PrintArray(b.ToArray());

            Tensor.Multiply(a, b, c);

            var result = c.ToArray();


           

            PrintArray(result);


            Console.ReadKey();




            //while (true)
            //{
            //    GpuNetBase test = new GpuNetBase();


            //    test.Test();
            //    Thread.Sleep(100);
            //}
            Console.WriteLine("end");
            Console.ReadKey();
        }

        static void PrintArray(float[,] array)
        {
            Console.WriteLine();
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write($"{array[i, j]} ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
        static void ILGPU()
        {
            ILGPUNet a = new ILGPUNet(new float[,]
            {
                {1, 2, 3},
                {1, 2, 3}
            });
            a.Width = 2;
            a.Height = 2;
            ILGPUNet b = new ILGPUNet(new float[,]
            {
                {1, 2, 3},
                {1, 2, 3}
            });
            b.Width = 2;
            b.Height = 1;
            ILGPUNet c = new ILGPUNet(new float[,]
            {
                {1, 2, 3},
                {1, 2, 3},
                {1, 2, 3}
            });
            c.Width = 2;
            c.Height = 1;

            //ILGPUNet.Add(a, b, c);

            ILGPUNet.Multiply(a, b, c);

            var result = c.ToArray();




            PrintArray(result);
        }
        static void GpuMemoryTest()
        {
            ILGPUNet a = new ILGPUNet(10000, 10000, 1F);
            ILGPUNet b = new ILGPUNet(10000, 10000, 1F);

            long start = Stopwatch.GetTimestamp();

            for (int i = 0; i < 10000000; i++)
            {
                b.AddToA(a);
                // using var result = b.Add(b);

                if (i % 100 == 0)
                {
                    long end = Stopwatch.GetTimestamp();

                    Console.WriteLine($"{ Math.Round(((end - start) * 1000.0 / Stopwatch.Frequency), 2)} ms");
                    start = Stopwatch.GetTimestamp();
                }
            }
        }
    }
}
