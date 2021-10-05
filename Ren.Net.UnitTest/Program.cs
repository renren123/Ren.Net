using Ren.Net.Objects;
using System.Collections.Generic;
using System;
using Ren.Net.Fully.Network;
using Ren.Net.ActivationFunction;

namespace Ren.Net.UnitTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Sequential layer = new Sequential(new List<NetModule>() 
            {
                new Linear(10, 10),
                new ReLU()
            });

            Console.WriteLine("Hello World!");

            Console.ReadKey();
        }
    }
}
