﻿using MathNet.Numerics.LinearAlgebra;
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
    }
}
