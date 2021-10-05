using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Util
{
    static class ClassPublicValue
    {
        public static bool isReadyToUbAndSigmma { set; get; }
        public static string TrainOrTest { set; get; }
        public static volatile bool processBarClose = false;
        /// <summary>
        /// 将静态变量进栈
        /// </summary>
        /// <param name="stack"></param>
        /// <returns></returns>
        public static Stack<object> SetStaticValue(Stack<object> stack)
        {
            if (stack == null)
                stack = new Stack<object>();
            stack.Push(isReadyToUbAndSigmma);
            stack.Push(TrainOrTest);
            return stack;
        }
        /// <summary>
        /// 将静态变量出栈
        /// </summary>
        /// <param name="stack"></param>
        public static void GetStaticValue(Stack<object> stack)
        {
            if (stack == null || stack.Count == 0)
            {
                Console.WriteLine("BNElement->GetStaticValue()");
                return;
            }
            TrainOrTest = (string)stack.Pop();
            isReadyToUbAndSigmma = (bool)stack.Pop();

        }
    }
}
