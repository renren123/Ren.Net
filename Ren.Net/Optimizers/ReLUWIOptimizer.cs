using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Optimizers
{
    [Serializable]
    public class ReLUWIOptimizer : WIInitialization
    {
        private static readonly Random r = new Random(DateTime.UtcNow.Millisecond);
        public override float GetWI(int sumInput)
        {
            float y = (float)r.NextDouble();
            float x = (float)r.NextDouble();
            float number = (float)(Math.Cos(2 * Math.PI * x) * Math.Sqrt(-2 * Math.Log(1 - y)));
            number *= (float)Math.Sqrt(2.0 / sumInput);
            number = Math.Abs(number);
            number *= (2.0F / sumInput);

            return number;
        }
    }
}
