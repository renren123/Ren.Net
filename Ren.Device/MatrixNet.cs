using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Device
{
    public class MatrixNet : DeviceNetBase
    {
        public override int Column { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public override int Row { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

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
        public MatrixNet(int m, int n)
        {
            Data = MBuild.Dense(m, n, 0F);
        }
        public MatrixNet(int m, int n, float value)
        {
            Data = MBuild.Dense(m, n, value);
        }
        public MatrixNet(int m, int n, Func<int, int, float> init)
        {
            Data = MBuild.Dense(m, n, init);
        }
        public override object Clone()
        {
            return new MatrixNet(Data.Clone());
        }
        public override float RowAverage(int i)
        {
            return Data.Row(i).Average();
        }
        public override float ColumnAverage(int j)
        {
            return Data.Column(j).Average();
        }
        public override float RowVariance(int index)
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
        public override float ColumnVariance(int index)
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
        public override void AddOneColumnWithValue(int length, float value)
        {
            Vector<float> vector = VBuild.Dense(length, value);
            this.Data = this.Data.InsertColumn(Column, vector);
        }
        /// <summary>
        /// 增加一行 赋值为 value
        /// </summary>
        /// <param name="length"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public override void AddOneRowWithValue(int length, float value)
        {
            Vector<float> vector = VBuild.Dense(length, value);
            this.Data = this.Data.InsertRow(Row, vector);
        }
        public override void RemoveLastOneColumn()
        {
            this.Data = this.Data.RemoveColumn(Column - 1);
        }
        public override void RemoveLastOneRow()
        {
            this.Data = this.Data.RemoveRow(Row - 1);
        }
        /// <summary>
        /// 增加一列，加到最后
        /// </summary>
        /// <param name="column"></param>
        public override void AddColumn(float[] column)
        {
            Vector<float> vector = VBuild.Dense(column);
            this.Data = this.Data.InsertColumn(Column, vector);
        }
        public override void AddRow(float[] column)
        {
            Vector<float> vector = VBuild.Dense(column);
            this.Data = this.Data.InsertRow(Row, vector);
        }
        public override void InsertColumn(int columnIndex, float[] column)
        {
            Vector<float> vector = VBuild.Dense(column);
            this.Data.InsertColumn(columnIndex, vector);
        }
        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <returns></returns>
        public override void Transpose()
        {
            this.Data = this.Data.Transpose();
        }
        /// <summary>
        /// 获取具体数值（待确定 pytorch 是不是同样的作用）
        /// </summary>
        /// <returns></returns>
        public override float GetItem()
        {
            return this.Data.RowSums().Sum() / (Row * Column);
        }
        /// <summary>
        /// 矩阵点乘，对应位相乘
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public override void DotMultiply(DeviceNetBase data)
        {
            MatrixNet matrixNet = data as MatrixNet;
            this.Data = Matrix<float>.op_DotDivide(this.Data, matrixNet.Data);
        }
        /// <summary>
        /// 开方
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public override void Sqrt()
        {
            this.Data = Matrix<float>.Sqrt(this.Data);
        }
        /// <summary>
        /// 点除，对应位相除
        /// </summary>
        /// <param name="dividend"></param>
        /// <param name="divisor"></param>
        /// <returns></returns>
        public override void DotDivide(DeviceNetBase divisor)
        {
            MatrixNet matrixNet = divisor as MatrixNet;
            this.Data = Matrix<float>.op_DotDivide(this.Data, matrixNet.Data);
        }
        public override float this[int i, int j] { get => Data[i, j]; set => Data[i, j] = value; }


        public static MatrixNet operator *(MatrixNet lhs, MatrixNet rhs)
        {
            throw new NotImplementedException();
        }
        public static MatrixNet operator *(float lhs, MatrixNet rhs)
        {
            throw new NotImplementedException();
        }
        public static MatrixNet operator *(MatrixNet lhs, float rhs)
        {
            throw new NotImplementedException();
        }
        public static MatrixNet operator /(MatrixNet lhs, float rhs)
        {
            throw new NotImplementedException();
        }
        public static MatrixNet operator +(MatrixNet lhs, MatrixNet rhs)
        {
            throw new NotImplementedException();
        }
        public static MatrixNet operator +(MatrixNet lhs, float rhs)
        {
            throw new NotImplementedException();
        }
        public static MatrixNet operator -(MatrixNet lhs, MatrixNet rhs)
        {
            throw new NotImplementedException();
        }
    }
}
