using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Extensions
{
    public static class FloatExtension
    {
        public static bool EqualsValue(this float left, float right, int decimals)
        {
            return Math.Round(left, decimals) == Math.Round(right, decimals);
        }
    }
}
