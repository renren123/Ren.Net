using Ren.Net.Objects;
using System.Collections.Generic;
using System;
using Ren.Net.Fully.Network;
using Ren.Net.ActivationFunction;

namespace Ren.Net.UnitTest
{
    public class Program
    {
        static void Main(string[] args)
        {
            Torch input = new Torch(10, 1);
            Torch output = null;

            Sequential netWork = new Sequential(new List<NetModule>()
            {
                // layer1
                new Linear(10, 10),
                new ReLU(),

                // layer2
                new Linear(10, 10),
                new ReLU(),
                // layer3
                new Linear(10, 1),
            });

            output = netWork.Forward(input);

            Console.WriteLine(output.Data);

            Console.WriteLine("Hello World!");

            Console.ReadKey();
        }
    }
}
