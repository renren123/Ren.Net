using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Extensions
{
    public static class ListExtension
    {
        /// <summary>
        /// 实现：https://www.skyfinder.cc/2020/04/01/csharprandomlist/
        /// 打乱 list 数据
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="sources"></param>
        public static void ListRandom<T>(this List<T> sources)
        {
            Random rd = new Random(System.Guid.NewGuid().GetHashCode());
            int index = 0;
            T temp;
            for (int i = 0; i < sources.Count; i++)
            {
                index = rd.Next(0, sources.Count - 1);
                if (index != i)
                {
                    temp = sources[i];
                    sources[i] = sources[index];
                    sources[index] = temp;
                }
            }
        }
    }
}
