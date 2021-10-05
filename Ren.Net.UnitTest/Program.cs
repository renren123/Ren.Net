using Ren.Net.Objects;
using System.Collections.Generic;
using System;
using Ren.Net.Fully.Network;
using Ren.Net.ActivationFunction;

namespace Ren.Net.UnitTest
{
    public class Program : NetManager
    {
        Sequential layer1 = new Sequential(new List<NetModule>()
            {
                new Linear(10, 10),
                new ReLU()
            });

        Sequential layer2 = new Sequential(new List<NetModule>()
            {
                new Linear(10, 10),
                new ReLU()
            });

        Sequential layer3 = new Sequential(new List<NetModule>()
            {
                new Linear(10, 1),
            });
        public override Torch Forward(Torch @in)
        {
            Torch output = null;

            output = layer1.Forward(@in);
            output = layer2.Forward(output);
            output = layer3.Forward(output);

            return output;
        }

        public override Torch Backup(Torch @out)
        {
            return base.Backup(@out);
        }


        static void Main(string[] args)
        {
            Torch input = null;
            Torch output = null;

            Console.WriteLine("Hello World!");

            Console.ReadKey();
        }
    }
}
