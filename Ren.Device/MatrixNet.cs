using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Device
{
    [Serializable]
    public class MatrixNet : DataInterface
    {
        public int Row => this.Data.RowCount;
        public int Column => this.Data.ColumnCount;
        public DeviceTpye Device { get; } = DeviceTpye.CUDA;

        private static MatrixBuilder<float> MBuild { get; } = Matrix<float>.Build;
        private static VectorBuilder<float> VBuild { get; } = Vector<float>.Build;
        private Matrix<float> Data { set; get; }

        private MatrixNet(Matrix<float> data)
        {
            this.Data = data;
        }
        public MatrixNet(float[,] data)
        {
            this.Data = MBuild.DenseOfArray(data);
        }
        public MatrixNet(int m, int n, float value)
        {
            Data = MBuild.Dense(m, n, value);
        }
        public MatrixNet(int m, int n, Func<int, int, float> init)
        {
            Data = MBuild.Dense(m, n, init);
        }
        public object Clone()
        {
            return new MatrixNet(Data.Clone());
        }
        public float RowAverage(int index)
        {
            return Data.Row(index).Average();
        }
        public float ColumnAverage(int j)
        {
            return Data.Column(j).Average();
        }
        public float RowVariance(int index)
        {
            var row = Data.Row(index);
            float sum = 0F;
            float average = row.Average();
            for (int i = 0; i < row.Count; i++)
            {
                sum += (row[i] - average) * (row[i] - average);
            }
            return sum / row.Count;
        }
        public float ColumnVariance(int index)
        {
            var column = Data.Column(index);
            float sum = 0F;
            float average = column.Average();
            for (int i = 0; i < column.Count; i++)
            {
                sum += (column[i] - average) * (column[i] - average);
            }
            return sum / column.Count;
        }
        public DataInterface AddOneColumnWithValue(int length, float value)
        {
            Vector<float> vector = VBuild.Dense(length, value);
            var data = this.Data.InsertColumn(Column, vector);
            return new MatrixNet(data);
        }
        /// <summary>
        /// 增加一行 赋值为 value
        /// </summary>
        /// <param name="length"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public DataInterface AddOneRowWithValue(int length, float value)
        {
            Vector<float> vector = VBuild.Dense(length, value);
            var data = this.Data.InsertRow(Row, vector);
            return new MatrixNet(data);
        }
        public DataInterface RemoveLastOneColumn()
        {
            var data = this.Data.RemoveColumn(Column - 1);
            return new MatrixNet(data);
        }
        public DataInterface RemoveLastOneRow()
        {
            var data = this.Data.RemoveRow(Row - 1);
            return new MatrixNet(data);
        }
        /// <summary>
        /// 增加一列，加到最后
        /// </summary>
        /// <param name="column"></param>
        //public void AddColumn(float[] column)
        //{
        //    Vector<float> vector = VBuild.Dense(column);
        //    this.Data = this.Data.InsertColumn(Column, vector);
        //}
        //public void AddRow(float[] column)
        //{
        //    Vector<float> vector = VBuild.Dense(column);
        //    this.Data = this.Data.InsertRow(Row, vector);
        //}
        //public void InsertColumn(int columnIndex, float[] column)
        //{
        //    Vector<float> vector = VBuild.Dense(column);
        //    this.Data.InsertColumn(columnIndex, vector);
        //}
        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <returns></returns>
        public DataInterface Transpose()
        {
            return new MatrixNet(this.Data.Transpose());
        }
        /// <summary>
        /// 获取具体数值（待确定 pytorch 是不是同样的作用）
        /// </summary>
        /// <returns></returns>
        public float GetItem()
        {
            return this.Data.RowSums().Sum() / (Row * Column);
        }
        /// <summary>
        /// 矩阵点乘，对应位相乘
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public DataInterface DotMultiply(DataInterface data)
        {
            MatrixNet matrixNet = data as MatrixNet;
            var copy = Matrix<float>.op_DotMultiply(this.Data, matrixNet.Data);
            return new MatrixNet(copy);
        }
        public DataInterface Multiply(DataInterface rhs)
        {
            return new MatrixNet(this.Data * (rhs as MatrixNet).Data);
        }
        /// <summary>
        /// 开方
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public DataInterface Sqrt()
        {
            var result = Matrix<float>.Sqrt(this.Data);
            return new MatrixNet(result);
        }
        /// <summary>
        /// 点除，对应位相除
        /// </summary>
        /// <param name="dividend"></param>
        /// <param name="divisor"></param>
        /// <returns></returns>
        public DataInterface DotDivide(DataInterface divisor)
        {
            MatrixNet matrixNet = divisor as MatrixNet;
            var result = Matrix<float>.op_DotDivide(this.Data, matrixNet.Data);
            return new MatrixNet(result);
        }

        public DataInterface Multiply(float rhs)
        {
            return new MatrixNet(this.Data * rhs);
        }

        public DataInterface Divide(float rhs)
        {
            return new MatrixNet(this.Data / rhs);
        }

        public DataInterface Add(DataInterface rhs)
        {
            MatrixNet matrixNet = rhs as MatrixNet;
            return new MatrixNet(this.Data + matrixNet.Data);
        }

        public DataInterface Add(float rhs)
        {
            return new MatrixNet(this.Data + rhs);
        }

        public DataInterface Minus(DataInterface rhs)
        {
            MatrixNet matrixNet = rhs as MatrixNet;
            return new MatrixNet(this.Data - matrixNet.Data);
        }

        public DataInterface Relu(DataInterface old)
        {
            MatrixNet oldData = old as MatrixNet;
            Matrix<float> newData = Data.Clone();
            for (int i = 0; i < Row; i++)
            {
                for (int j = 0; j < Column; j++)
                {
                    if(oldData[i, j] < 0F)
                    {
                        newData[i, j] = 0F;
                    }
                }
            }
            return new MatrixNet(newData);
        }

        public void Dispose()
        {
            this.Data = null;
        }

        public void AddToA(DataInterface rhs)
        {
            MatrixNet matrixNet = rhs as MatrixNet;
            this.Data = Data + matrixNet.Data;
        }

        public void MinusToA(DataInterface rhs)
        {
            MatrixNet matrixNet = rhs as MatrixNet;
            this.Data = Data - matrixNet.Data;
        }

        public float this[int i, int j] { get => Data[i, j]; set => Data[i, j] = value; }
    }
}
