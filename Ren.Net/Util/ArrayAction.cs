using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Util
{
    public class ArrayAction
    {
        /// <summary>
        /// 输入的是minibatch个样本的数量，第二个list是一个filter里面wi的数量
        /// 目的求得是他们的均值，即一个filter里面Wi数量的均值
        /// </summary>
        /// <param name="list"></param>
        /// <returns></returns>
        public static List<float[,]> ListAverage(List<List<float[,]>> list)
        {
            List<float[,]> averageArray = new List<float[,]>(list[0].Count);
            for (int i = 0; i < list[0].Count; i++)
            {
                averageArray.Add(new float[list[0][0].GetLength(0), list[0][0].GetLength(1)]);
            }
            for (int i = 0; i < averageArray.Count; i++)
            {
                for (int x = 0; x < averageArray[i].GetLength(0); x++)
                {
                    for (int y = 0; y < averageArray[i].GetLength(1); y++)
                    {
                        float sum = 0;
                        int listCount = list.Count;
                        for (int countList = 0; countList < listCount; countList++)
                        {
                            sum += list[countList][i][x, y];
                        }
                        averageArray[i][x, y] = sum / listCount;
                    }
                }
            }
            return averageArray;
        }
        /// <summary>
        /// 均值
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static float average(float[] x)
        {
            float sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                sum += x[i];
            }
            return sum / x.Length;
        }
        /// <summary>
        /// 标准方差
        /// </summary>
        /// <param name="x"></param>
        /// <param name="average"></param>
        /// <returns></returns>
        public static float variance(float[] x, float average)
        {
            float sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                sum += (x[i] - average) * (x[i] - average);
            }
            return sum / x.Length;
        }
    }
}
