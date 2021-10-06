using Ren.Net.Objects;
using System.Collections.Generic;
using System;
using Ren.Net.Fully.Network;
using Ren.Net.ActivationFunction;
using Ren.Net.Loss;

namespace Ren.Net.UnitTest
{
    public class Program
    {
        static void Main(string[] args)
        {
            // 模拟 函数 y = x + 1
            Torch input = new Torch() 
            {
                Data = new List<float[]>(1)
            };
            input.Data.Add(new float[] { 1 });

            Torch label = new Torch()
            {
                Data = new List<float[]>(1)
            };
            label.Data.Add(new float[] { 2 });

            Torch output = null;

            Sequential netWork = new Sequential(new List<NetModule>()
            {
                // layer1
                new Linear(1, 10),
                new ReLU(),
                // layer2
                new Linear(10, 10),
                new ReLU(),
                // layer3
                new Linear(10, 1),
            });
            MSELoss loss = new MSELoss();

            for (int i = 0; i < 1000; i++)
            {
                output = netWork.Forward(input);

                Console.WriteLine(output.Data[0][0]);

                var sensitive = loss.CaculateLoss(label, output);

                netWork.Backup(sensitive);
            }

            

            Console.WriteLine(output.Data);

            Console.WriteLine("Hello World!");

            Console.ReadKey();
        }
    }
}
