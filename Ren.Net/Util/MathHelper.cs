using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Util
{
    public static class MathHelper
    {
        public static int Max(int a, int b, int c)
        {
            return Math.Max(Math.Max(a, b), c);
        }
    }
}
