using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Ren.Net.Test
{
    class MathNetTest
    {
        static object lockOB = new object(); 

        static void Main(string[] args)
        {
            //Dictionary<int, int> keys = new Dictionary<int, int>();
            //Parallel.For(0, 1000, (xp) =>
            //{
            //    lock (lockOB)
            //    {
            //        Thread.Sleep(1);
            //    }

            //    lock (keys)
            //    {
            //        if(!keys.ContainsKey(xp))
            //        {
            //            keys[xp] = 1;
            //        }
            //        else
            //        {
            //            keys[xp] += 1;
            //        }
            //    }

            //});
            //var orderKeys = keys.OrderBy(p => p.Key).ToDictionary(p=>p.Key, p=>p.Value);
            //int count = 0;
            //foreach (var item in orderKeys)
            //{
            //    if(item.Value == 0 || item.Value > 1)
            //    {
            //        int a = 0;
            //    }
            //    if (item.Key == count)
            //    {
            //        count++;
            //    }
            //    else
            //    {
            //        break;
            //    }
            //}


            float[,] testData1 = new float[2, 2] 
            { 
                { 1F, 2F },
                { 3F, 4F },
            };
            float[,] testData2 = new float[2, 2]
            {
                { 2F, 2F },
                { 3F, 5F },
            };

            var mb = Matrix<float>.Build;
            var vb = Vector<float>.Build;

            Matrix<float> a = mb.DenseOfArray(testData1);
            Matrix<float> b = mb.DenseOfArray(testData2);


            Matrix<float> c = mb.Dense(5000, 5000, 1F);
            Matrix<float> d = mb.Dense(5000, 5000, 1F);


            //CreateVector.Dense()

            for (int i = 0; i < 10000; i++)
            {
                var result = c * d;
           
                Console.WriteLine(i);
            }



            Console.ReadKey();



            //Vector<float> vector = vb.Dense(new float[] { 100F, 100F});
            //// vector Variance

            //var sqrtTest = Matrix<float>.Sqrt(matrix2);
            //var dotTest = Matrix<float>.op_DotMultiply(matrix2, matrix2);
            //var devide = Matrix<float>.op_DotDivide(matrix1, matrix2);

            //Console.WriteLine(devide.ToString());
            //Console.WriteLine(sqrtTest.ToString());
            //Console.WriteLine(dotTest.ToString());

            //var temp = matrix1.InsertColumn(2, vector);

            //Console.WriteLine("\r\n" + temp.ToString());

            //var result = (matrix1 * matrix2).ToArray();

            //var avereage = matrix1;
            //var addTest = matrix1.Row(0).Add(100);

            //// ar test = avereage + addTest;

            //Console.ReadKey();
            //var mb = Matrix<float>.Build;
            //Matrix<float> fromArray = mb.DenseOfArray(testData);

            //var B = new DenseMatrix(new double[,]
            //{
            //    { 1, 1, 2}
            //});

            //var a = new DenseMatrix(testData);
        }
    }
}
