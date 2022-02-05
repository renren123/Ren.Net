using System;
using System.Collections.Generic;
using System.Text;
using Ren.Net.Objects;

namespace Ren.Net.Extensions
{
    public static class TensorExtension
    {
        /// <summary>
        /// 判断两个 tensor 是不是值相等
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static bool EqualsValue(this Tensor left, Tensor right, int decimals)
        {
            if (left.Row == 0 || left.Column == 0 || right.Row == 0 || right.Column == 0)
            {
                return false;
            }
            if (left.Row != right.Row || left.Column != right.Column)
            {
                return false;
            }
            for (int i = 0; i < left.Row; i++)
            {
                for (int j = 0; j < left.Column; j++)
                {
                    if (!left[i, j].EqualsValue(right[i, j], decimals))
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        public static void PrintArray(this Tensor tensor)
        {
            var array = tensor.ToArray();
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
