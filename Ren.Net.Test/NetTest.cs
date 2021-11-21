using Ren.Net.Networks;
using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ren.Net.Test
{
    class NetTest
    {
        static void Main(string[] args)
        {
            NetModule bn = new BatchNorm1DCPU(2);

            Tensor.Device = Device.DeviceTpye.CPU;
            bn.Init();

            //Tensor input = new Tensor(new float[3, 2]{
            //    { 1F, 2F },
            //    { 1F, 3F },
            //    { 1F, 4F }
            //});
            //Tensor label = new Tensor(new float[3, 2]{
            //    { 2F, 2F },
            //    { 1F, 1F },
            //    { 1F, 1F }
            //});

            Tensor input = new Tensor(new float[2, 3]{
                { 1F, 1F, 1F },
                { 2F, 3F, 4F },
            });
            Tensor label = new Tensor(new float[2, 3]{
                { 2F, 1F, 1F },
                { 2F, 1F, 1F },
            });

            for (int i = 0; i < 2; i++)
            {
                var output = bn.Forward(input);
                var dout = bn.Backup(label);

                PrintArray(output.ToArray());
                PrintArray(dout.ToArray());
                Console.WriteLine("########################");
            }

            

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
    }
}
