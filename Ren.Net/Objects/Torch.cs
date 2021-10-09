using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Objects
{
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
        public Torch() { }
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
        public Torch(int neuronNumber, int batch, Func<int, int, float> init)
        {
            Data = MBuild.Dense(neuronNumber, batch, init);
        }
        public object Clone()
        {
            return new Torch(Data.Clone());
        }
        public static Torch operator *(Torch lhs, Torch rhs)
        {
            return new Torch(lhs.Data * rhs.Data);
        }
        public static Torch operator +(Torch lhs, Torch rhs)
        {
            return new Torch(lhs.Data + rhs.Data);
        }
        public static Torch operator -(Torch lhs, Torch rhs)
        {
            return new Torch(lhs.Data - rhs.Data);
        }
        public float RowAverage(int i)
        {
            return Data.Row(i).Average();
        }
        public Torch AddOneColumnWithValue(int length, float value)
        {
            Vector<float> vector = VBuild.Dense(length, value);
            var data = this.Data.InsertColumn(Column, vector);
            return new Torch(data);
        }
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
        public override string ToString()
        {
            return this.Data.ToMatrixString();
        }
    }
}
