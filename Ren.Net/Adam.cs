using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net
{
    class Adam
    {
        /// <summary>
        /// 这两个数用于保存B1与B2的次方
        /// </summary>
        public static double B1_pow = 1;
        /// <summary>
        /// 这两个数用于保存B1与B2的次方
        /// </summary>
        public static double B2_pow = 1;
        private static int t_step = 1;
        public static double U_pow = 1;
        public static double V_pow = 1;
        private static double u = 0.9;
        private static double v = 0.999;
        public static float E { set; get; } = 0.00000001F;

        public float Vgamma { set; get; }
        public float Sgamma { set; get; }
        public Adam()
        {
            Vgamma = Sgamma = 0;
        }
        /// <summary>
        /// 更新一个数值
        /// </summary>
        /// <param name="dgamma">数据的增量，数据的变化值</param>
        /// <returns></returns>
        public float GetAdam(float dgamma)
        {
            Vgamma = (float)(AgentClass.B1 * Vgamma + (1 - AgentClass.B1) * dgamma);
            Sgamma = (float)(AgentClass.B2 * Sgamma + (1 - AgentClass.B2) * dgamma * dgamma);
            float Vgamma_correction = (float)(Vgamma / (1 - Adam.B1_pow));
            float Sgamma_correction = (float)(Sgamma / (1 - Adam.B2_pow));

            return (float)(AgentClass.Study_rate * Vgamma_correction / (Math.Sqrt(Sgamma_correction) + Adam.E));
        }
        public static double GetAdamNumber(double m_old, double n_old, double g_t, out double m_new, out double n_new)
        {
            m_new = U * m_old + (1 - U) * g_t;
            n_new = V * n_old + (1 - V) * g_t * g_t;
            U_pow *= U;
            V_pow *= V;
            double m_average = m_new / (1 - U_pow);
            double n_average = n_new / (1 - V_pow);
            return m_average / (Math.Sqrt(n_average) + E);
        }

        public static Stack<object> SetStaticValue(Stack<object> stack)
        {
            if (stack == null)
                stack = new Stack<object>();
            stack.Push(B1_pow);
            stack.Push(B2_pow);
            stack.Push(U_pow);
            stack.Push(V_pow);
            stack.Push(T_step);
            return stack;
        }
        public static void GetStackValue(Stack<object> stack)
        {
            if (stack == null)
            {
                Console.WriteLine("Adam->GetStackValue()");
            }
            T_step = (int)stack.Pop();
            V_pow = (double)stack.Pop();
            U_pow = (double)stack.Pop();
            B2_pow = (double)stack.Pop();
            B1_pow = (double)stack.Pop();
        }

        public static int T_step
        {
            get
            {
                return t_step;
            }

            set
            {
                t_step = value;
            }
        }

        public static double U
        {
            get
            {
                return u;
            }

            set
            {
                u = value;
            }
        }

        public static double V
        {
            get
            {
                return v;
            }

            set
            {
                v = value;
            }
        }
    }
}
