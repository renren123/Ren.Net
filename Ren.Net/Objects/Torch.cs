using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Objects
{
    [Serializable]
    public class Torch : ICloneable
    {
        /// <summary>
        /// 几个神经元 batch 数据，list 的长度 是一层神经元的数量，float 是 batch 的大小
        /// 行数 是神经元的数量，列数是 batchsize 的数量
        /// </summary>
        private Matrix<float> Data { set; get; }
        public int Column => Data == null || Data.RowCount == 0 ? -1 : Data.ColumnCount;
        public int Row => Data == null || Data.RowCount == 0 ? -1 : Data.RowCount;
        private static MatrixBuilder<float> MBuild { get; } = Matrix<float>.Build;
        private static VectorBuilder<float> VBuild { get; } = Vector<float>.Build;
        public Torch(float[,] data)
        {
            this.Data = MBuild.DenseOfArray(data);
        }
        private Torch(Matrix<float> data)
        {
            this.Data = data;
        }
        /// <summary>
        /// 初始化 一个 torch
        /// </summary>
        /// <param name="neuronNumber">神经元的数量</param>
        /// <param name="batch">一个 batch 的大小</param>
        public Torch(int neuronNumber, int batch)
        {
            Data = MBuild.Dense(neuronNumber, batch, 0F);
        }
        public Torch(int neuronNumber, int batch, float value)
        {
            Data = MBuild.Dense(neuronNumber, batch, value);
        }
        public Torch(int neuronNumber, int batch, Func<int, int, float> init)
        {
            Data = MBuild.Dense(neuronNumber, batch, init);
        }
        public object Clone()
        {
            return new Torch(Data.Clone());
        }
        
        public float RowAverage(int i)
        {
            return Data.Row(i).Average();
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
        public Torch AddOneColumnWithValue(int length, float value)
        {
            Vector<float> vector = VBuild.Dense(length, value);
            var data = this.Data.InsertColumn(Column, vector);
            return new Torch(data);
        }
        /// <summary>
        /// 增加一行 赋值为 value
        /// </summary>
        /// <param name="length"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public Torch AddOneRowWithValue(int length, float value)
        {
            Vector<float> vector = VBuild.Dense(length, value);
            var data = this.Data.InsertRow(Row, vector);
            return new Torch(data);
        }
        public Torch RemoveLastOneColumn()
        {
            var data = this.Data.RemoveColumn(Column - 1);
            return new Torch(data);
        }
        public Torch RemoveLastOneRow()
        {
            var data = this.Data.RemoveRow(Row - 1);
            return new Torch(data);
        }
        /// <summary>
        /// 增加一列，加到最后
        /// </summary>
        /// <param name="column"></param>
        public void AddColumn(float[] column)
        {
            Vector<float> vector = VBuild.Dense(column);
            this.Data = this.Data.InsertColumn(Column, vector);
        }
        public void AddRow(float[] column)
        {
            Vector<float> vector = VBuild.Dense(column);
            this.Data = this.Data.InsertRow(Row, vector);
        }
        public void InsertColumn(int columnIndex, float[] column)
        {
            Vector<float> vector = VBuild.Dense(column);
            this.Data.InsertColumn(columnIndex, vector);
        }
        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <returns></returns>
        public Torch Transpose()
        {
            return new Torch(this.Data.Transpose());
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
        public static Torch DotMultiply(Torch a, Torch b)
        {
            return new Torch(Matrix<float>.op_DotMultiply(a.Data, b.Data));
        }
        /// <summary>
        /// 开方
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Torch Sqrt(Torch a)
        {
            return new Torch(Matrix<float>.Sqrt(a.Data));
        }
        /// <summary>
        /// 点除，对应位相除
        /// </summary>
        /// <param name="dividend"></param>
        /// <param name="divisor"></param>
        /// <returns></returns>
        public static Torch DotDivide(Torch dividend, Torch divisor)
        {
            return new Torch(Matrix<float>.op_DotDivide(dividend.Data, divisor.Data));
        }
        public override string ToString()
        {
            return this.Data.ToMatrixString();
        }
        public override bool Equals(object obj)
        {
            if (!(obj is Torch))
            {
                return false;
            }
            Torch torch = obj as Torch;

            if (torch.Row != this.Row || torch.Column != this.Column)
            {
                return false;
            }
            for (int i = 0; i < torch.Row; i++)
            {
                for (int j = 0; j < torch.Column; j++)
                {
                    if(torch[i, j] != this[i, j])
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        public override int GetHashCode()
        {
            return Data.GetHashCode();
        }

        /// <summary>
        /// torch 索引器
        /// </summary>
        /// <param name="i">行数</param>
        /// <param name="j">列数</param>
        /// <returns></returns>
        public float this[int i, int j]
        {
            get
            {
                return this.Data[i, j];
            }
            set
            {
                this.Data[i, j] = value;
            }
        }
        public static Torch operator *(Torch lhs, Torch rhs)
        {
            return new Torch(lhs.Data * rhs.Data);
        }
        public static Torch operator *(float lhs, Torch rhs)
        {
            return new Torch(lhs * rhs.Data);
        }
        public static Torch operator *(Torch lhs, float rhs)
        {
            return new Torch(lhs.Data * rhs);
        }
        public static Torch operator /(Torch lhs, float rhs)
        {
            return new Torch(lhs.Data / rhs);
        }

        public static Torch operator +(Torch lhs, Torch rhs)
        {
            return new Torch(lhs.Data + rhs.Data);
        }
        public static Torch operator +(Torch lhs, float rhs)
        {
            return new Torch(lhs.Data + rhs);
        }
        public static Torch operator -(Torch lhs, Torch rhs)
        {
            return new Torch(lhs.Data - rhs.Data);
        }
    }
}
