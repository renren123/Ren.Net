using Ren.Net.Objects;
using System.Collections.Generic;
using System;
using Ren.Net.Fully.Network;
using Ren.Net.ActivationFunction;
using Ren.Net.Loss;
using Ren.Net.Optimizers;

namespace Ren.Net.UnitTest
{
    public class Program
    {
        static void Main(string[] args)
        {
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
            netWork.Optimizer = new Adam(learningRate: 0.001F);

            MSELoss loss = new MSELoss();

            for (int i = 0; i < 10000; i++)
            {
                var (input, label) = GetTorch();

                output = netWork.Forward(input);

                var sensitive = loss.CaculateLoss(label, output);

                Console.WriteLine(sensitive.Data[0][0]);

                netWork.Backup(sensitive);
            }
            Console.WriteLine("END");

            Console.ReadKey();
        }
        /// <summary>
        /// 模拟 函数 y = x + 1
        /// </summary>
        /// <returns></returns>
        static (Torch input, Torch label) GetTorch()
        {
            int x = new Random().Next(1, 100);
            Torch input = new Torch()
            {
                Data = new List<float[]>(1)
            };
            input.Data.Add(new float[] { x });

            Torch label = new Torch()
            {
                Data = new List<float[]>(1)
            };
            label.Data.Add(new float[] { x + 1 });

            return (input, label);
        }
    }
}
