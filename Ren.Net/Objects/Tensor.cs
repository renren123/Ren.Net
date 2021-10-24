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
        public DeviceTpye Device { set; get; } = DeviceTpye.CPU;
        /// <summary>
        /// 几个神经元 batch 数据，list 的长度 是一层神经元的数量，float 是 batch 的大小
        /// 行数 是神经元的数量，列数是 batchsize 的数量
        /// </summary>
        private DataInterface deviceData { set; get; }
        public int Row => deviceData.Row;
        public int Column => deviceData.Column;
        private Tensor(DataInterface deviceData)
        {
            this.deviceData = deviceData;
        }
        public Tensor(float[,] data)
        {
            switch (Device)
            {
                case DeviceTpye.CPU:
                    deviceData = new MatrixNet(data);
                    break;
                case DeviceTpye.CUDA:
                    deviceData = new ILGPUNet(data);
                    break;
                case DeviceTpye.Default:
                    throw new Exception();
            }
        }
        /// <summary>
        /// 初始化 一个 torch
        /// </summary>
        /// <param name="neuronNumber">神经元的数量</param>
        /// <param name="batch">一个 batch 的大小</param>
        public Tensor(int neuronNumber, int batch)
        {
            switch (Device)
            {
                case DeviceTpye.CPU:
                    deviceData = new MatrixNet(neuronNumber, batch);
                    break;
                case DeviceTpye.CUDA:
                    deviceData = new ILGPUNet(neuronNumber, batch);
                    break;
                case DeviceTpye.Default:
                    throw new Exception();
            }
        }
        public Tensor(int neuronNumber, int batch, float value)
        {
            switch (Device)
            {
                case DeviceTpye.CPU:
                    deviceData = new MatrixNet(neuronNumber, batch, value);
                    break;
                case DeviceTpye.CUDA:
                    deviceData = new ILGPUNet(neuronNumber, batch, value);
                    break;
                case DeviceTpye.Default:
                    throw new Exception();
            }
        }
        public Tensor(int neuronNumber, int batch, Func<int, int, float> init)
        {
            switch (Device)
            {
                case DeviceTpye.CPU:
                    deviceData = new MatrixNet(neuronNumber, batch, init);
                    break;
                case DeviceTpye.CUDA:
                    deviceData = new ILGPUNet(neuronNumber, batch, init);
                    break;
                case DeviceTpye.Default:
                    throw new Exception();
            }
        }
        public object Clone()
        {
            return new Tensor(deviceData.Clone() as DataInterface);
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
            return new Tensor(deviceData.AddOneColumnWithValue(length, value));
        }
        /// <summary>
        /// 增加一行 赋值为 value
        /// </summary>
        /// <param name="length"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public Tensor AddOneRowWithValue(int length, float value)
        {
            return new Tensor(deviceData.AddOneRowWithValue(length, value));
        }
        public Tensor RemoveLastOneColumn()
        {
            return new Tensor(deviceData.RemoveLastOneColumn());
        }
        public Tensor RemoveLastOneRow()
        {
            return new Tensor(deviceData.RemoveLastOneRow());
        }
        /// <summary>
        /// 增加一列，加到最后
        /// </summary>
        /// <param name="column"></param>
        //public void AddColumn(float[] column)
        //{
        //    deviceData.AddColumn(column);
        //}
        //public void AddRow(float[] column)
        //{
        //    deviceData.AddRow(column);
        //}
        //public void InsertColumn(int columnIndex, float[] column)
        //{
        //    deviceData.InsertColumn(columnIndex, column);
        //}
        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <returns></returns>
        public Tensor Transpose()
        {
            return new Tensor(deviceData.Transpose());
        }
        public Tensor Relu(Tensor old)
        {
            return new Tensor(deviceData.Relu(old.deviceData));
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
            var result = a.deviceData.DotMultiply(b.deviceData);
            return new Tensor(result);
        }
        /// <summary>
        /// 开方
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static Tensor Sqrt(Tensor a)
        {
            var result = a.deviceData.Sqrt();
            return new Tensor(result);
        }
        /// <summary>
        /// 点除，对应位相除
        /// </summary>
        /// <param name="dividend"></param>
        /// <param name="divisor"></param>
        /// <returns></returns>
        public static Tensor DotDivide(Tensor dividend, Tensor divisor)
        {
            var result = dividend.deviceData.DotDivide(divisor.deviceData);
            return new Tensor(result);
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
