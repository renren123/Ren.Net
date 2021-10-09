using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ren.Net.Test
{
    class MathNetTest
    {
        static void Main(string[] args)
        {
            float[,] testData1 = new float[2, 2] 
            { 
                { 1F, 2F },
                { 3F, 4F },
            };
            float[,] testData2 = new float[2, 2]
            {
                { 1F, 2F },
                { 3F, 4F },
            };
            var mb = Matrix<float>.Build;
            var vb = Vector<float>.Build;

            Matrix<float> matrix1 = mb.DenseOfArray(testData1);
            Matrix<float> matrix2 = mb.DenseOfArray(testData2);

            matrix1.Transpose();

            Vector<float> vector = vb.Dense(new float[] { 100F, 100F});

            var temp = matrix1.InsertColumn(2, vector);

            Console.WriteLine("\r\n" + temp.ToString());

            var result = (matrix1 * matrix2).ToArray();

            var avereage = matrix1;
            var addTest = matrix1.Row(0).Add(100);

            // ar test = avereage + addTest;

            Console.ReadKey();
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
