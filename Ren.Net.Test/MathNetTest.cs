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

            Matrix<float> matrix1 = mb.DenseOfArray(testData1);
            Matrix<float> matrix2 = mb.DenseOfArray(testData2);

            var result = (matrix1 * matrix2).ToArray();

            matrix1.Row(0).Average();


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
