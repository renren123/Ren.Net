using System;
using System.Collections.Generic;
using System.Linq;
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
        /// <summary>
        /// 按指定数量均分, 实现：
        ///  https://stackoverflow.com/questions/30247968/splitting-a-list-or-collection-into-chunks
        ///  https://www.coder.work/article/3056584
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="me"></param>
        /// <param name="size"></param>
        /// <returns></returns>
        public static List<List<T>> SplitList<T>(this List<T> me, int size = 50)
        {
            var list = new List<List<T>>();
            for (int i = 0; i < me.Count; i += size)
                list.Add(me.GetRange(i, Math.Min(size, me.Count - i)));
            return list;
        }
    }
}
