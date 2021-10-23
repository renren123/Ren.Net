using MathNet.Numerics.LinearAlgebra;
using Ren.Device;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Objects
{
    [Serializable]
    public class Tensor : ICloneable
    {
        /// <summary>
        /// 几个神经元 batch 数据，list 的长度 是一层神经元的数量，float 是 batch 的大小
        /// 行数 是神经元的数量，列数是 batchsize 的数量
        /// </summary>
        private DeviceNetBase deviceData { set; get; }
        public int Column => deviceData.Column;
        public int Row => deviceData.Row;

        public Tensor(float[,] data)
        {
            deviceData = new DeviceNetBase(data);
        }
        private Tensor(DeviceNetBase deviceData)
        {
            this.deviceData = deviceData;
        }
        /// <summary>
        /// 初始化 一个 torch
        /// </summary>
        /// <param name="neuronNumber">神经元的数量</param>
        /// <param name="batch">一个 batch 的大小</param>
        public Tensor(int neuronNumber, int batch)
        {
            deviceData = new MatrixNet(neuronNumber, batch);
        }
        public Tensor(int neuronNumber, int batch, float value)
        {
            deviceData = new MatrixNet(neuronNumber, batch, value);
        }
        public Tensor(int neuronNumber, int batch, Func<int, int, float> init)
        {
            deviceData = new MatrixNet(neuronNumber, batch, init);
        }
        public object Clone()
        {
            return new Tensor(deviceData.Clone() as DeviceNetBase);
        }
        
        public float RowAverage(int i)
        {
            return deviceData.RowAverage(i);
        }
        public float ColumnAverage(int j)
        {
            return deviceData.ColumnAverage(j);
        }
        public float RowVariance(int index)
        {
            return deviceData.RowVariance(index);
        }
        public float ColumnVariance(int index)
        {
            return deviceData.ColumnVariance(index);
        }
        public Tensor AddOneColumnWithValue(int length, float value)
        {
            deviceData.AddOneColumnWithValue(length, value);
            return this;
        }
        /// <summary>
        /// 增加一行 赋值为 value
        /// </summary>
        /// <param name="length"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public Tensor AddOneRowWithValue(int length, float value)
        {
            deviceData.AddOneRowWithValue(length, value);
            return this;
        }
        public Tensor RemoveLastOneColumn()
        {
            deviceData.RemoveLastOneColumn();
            return this;
        }
        public Tensor RemoveLastOneRow()
        {
            deviceData.RemoveLastOneRow();
            return this;
        }
        /// <summary>
        /// 增加一列，加到最后
        /// </summary>
        /// <param name="column"></param>
        public void AddColumn(float[] column)
        {
            deviceData.AddColumn(column);
        }
        public void AddRow(float[] column)
        {
            deviceData.AddRow(column);
        }
        public void InsertColumn(int columnIndex, float[] column)
        {
            deviceData.InsertColumn(columnIndex, column);
        }
        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <returns></returns>
        public Tensor Transpose()
        {
            deviceData.Transpose();
            return this;
        }
        /// <summary>
        /// 获取具体数值（待确定 pytorch 是不是同样的作用）
        /// </summary>
        /// <returns></returns>
        public float GetItem()
        {
            return deviceData.GetItem();
        }
        /// <summary>
        /// 矩阵点乘，对应位相乘
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Tensor DotMultiply(Tensor a, Tensor b)
        {
            a.deviceData.DotMultiply(b.deviceData);
            return new Tensor(a.deviceData);
        }
        /// <summary>
        /// 开方
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Tensor Sqrt(Tensor a)
        {
            a.deviceData.Sqrt();
            return new Tensor(a.deviceData);
        }
        /// <summary>
        /// 点除，对应位相除
        /// </summary>
        /// <param name="dividend"></param>
        /// <param name="divisor"></param>
        /// <returns></returns>
        public static Tensor DotDivide(Tensor dividend, Tensor divisor)
        {
            dividend.deviceData.DotDivide(divisor.deviceData);
            return new Tensor(dividend.deviceData);
        }
        public override string ToString()
        {
            return this.deviceData.ToString();
        }
        public override bool Equals(object obj)
        {
            if (!(obj is Tensor))
            {
                return false;
            }
            Tensor torch = obj as Tensor;

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
            return deviceData.GetHashCode();
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
                return this.deviceData[i, j];
            }
            set
            {
                this.deviceData[i, j] = value;
            }
        }
        public static Tensor operator *(Tensor lhs, Tensor rhs)
        {
            return new Tensor(lhs.deviceData * rhs.deviceData);
        }
        public static Tensor operator *(float lhs, Tensor rhs)
        {
            return new Tensor(lhs * rhs.deviceData);
        }
        public static Tensor operator *(Tensor lhs, float rhs)
        {
            return new Tensor(lhs.deviceData * rhs);
        }
        public static Tensor operator /(Tensor lhs, float rhs)
        {
            return new Tensor(lhs.deviceData / rhs);
        }
        public static Tensor operator +(Tensor lhs, Tensor rhs)
        {
            return new Tensor(lhs.deviceData + rhs.deviceData);
        }
        public static Tensor operator +(Tensor lhs, float rhs)
        {
            return new Tensor(lhs.deviceData + rhs);
        }
        public static Tensor operator -(Tensor lhs, Tensor rhs)
        {
            return new Tensor(lhs.deviceData - rhs.deviceData);
        }
    }
}
