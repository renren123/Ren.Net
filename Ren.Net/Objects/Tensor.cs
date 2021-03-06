using MathNet.Numerics.LinearAlgebra;
using Ren.Device;
using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Ren.Net.Objects
{
    [Serializable]
    public class Tensor : ICloneable, IDisposable
    {
        public static int Interval { get; } = 2;
        public static int MaxLinearNumber { set; get; }
        public static DeviceTpye Device { set; get; } = DeviceTpye.Default;
        /// <summary>
        /// 几个神经元 batch 数据，list 的长度 是一层神经元的数量，float 是 batch 的大小
        /// 行数 是神经元的数量，列数是 batchsize 的数量
        /// </summary>
        private DataInterface deviceData { set; get; }
        public int Row => deviceData.Row;
        public int Column => deviceData.Column;
        public int Width 
        { 
            set 
            {
                deviceData.Width = value;
            } 
            get => deviceData.Width; 
        }
        public int Height 
        {
            set
            {
                deviceData.Height = value;
            }
            get => deviceData.Height;
        }
        private Tensor(DataInterface deviceData)
        {
            this.deviceData = deviceData;
        }
        public Tensor(float[] data)
        {
            float[,] temp = null;

            temp = new float[data.Length, 1];

            for (int i = 0; i < data.Length; i++)
            {
                temp[i, 0] = data[i];
            }

            switch (Device)
            {
                case DeviceTpye.CPU:
                    {
                        deviceData = new MatrixNet(temp);
                    }
                    break;
                case DeviceTpye.CUDA:
                    {
                        int witdh = temp.GetLength(0);
                        int height = temp.GetLength(1);

                        MaxLinearNumber = MathHelper.Max(witdh + Interval, height + Interval, MaxLinearNumber);

                        deviceData = new ILGPUNet(MaxLinearNumber, MaxLinearNumber, (int i, int j) =>
                        {
                            if (i < witdh && j < height)
                            {
                                return temp[i, j];
                            }
                            else
                            {
                                return 0F;
                            }
                        });
                    }
                    break;
                default:
                    throw new Exception(Device.ToString());
            }
            deviceData.Width = temp.GetLength(0);
            deviceData.Height = temp.GetLength(1);
        }
        public Tensor(float[,] data)
        {
            switch (Device)
            {
                case DeviceTpye.CPU:
                    deviceData = new MatrixNet(data);
                    break;
                case DeviceTpye.CUDA:
                    {
                        int witdh = data.GetLength(0);
                        int height = data.GetLength(1);

                        MaxLinearNumber = MathHelper.Max(witdh + Interval, height + Interval, MaxLinearNumber);

                        deviceData  =  new ILGPUNet(MaxLinearNumber, MaxLinearNumber, (int i, int j) =>
                        {
                            if (i < witdh && j < height)
                            {
                                return data[i, j];
                            }
                            else
                            {
                                return 0F;
                            }
                        });
                    }
                    break;
                case DeviceTpye.Default:
                    throw new Exception();
            }
            deviceData.Width = data.GetLength(0);
            deviceData.Height = data.GetLength(1);
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
            deviceData.Width = neuronNumber;
            deviceData.Height = batch;
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
            deviceData.Width = neuronNumber;
            deviceData.Height = batch;
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
        public Tensor AddLastOneRowWithValue(float value)
        {
            return new Tensor(deviceData.AddOneRowWithValue(Row, value));
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

        public void MinusToA(Tensor right)
        {
            this.deviceData.MinusToA(right.deviceData);
        }
        public void AddToA(Tensor right)
        {
            this.deviceData.AddToA(right.deviceData);
        }
        public Tensor Sqrt()
        {
            return new Tensor(this.deviceData.Sqrt());
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
        //public override bool Equals(object obj)
        //{
        //    if (!(obj is Tensor))
        //    {
        //        return false;
        //    }
        //    Tensor torch = obj as Tensor;

        //    if (torch.Row != this.Row || torch.Column != this.Column)
        //    {
        //        return false;
        //    }
        //    for (int i = 0; i < torch.Row; i++)
        //    {
        //        for (int j = 0; j < torch.Column; j++)
        //        {
        //            if(torch[i, j] != this[i, j])
        //            {
        //                return false;
        //            }
        //        }
        //    }
        //    return true;
        //}
        //public override int GetHashCode()
        //{
        //    return deviceData.GetHashCode();
        //}

        public float[,] ToArray()
        {
            return this.deviceData.ToArray();
        }

        public void Dispose()
        {
            this.deviceData.Dispose();
        }
        /// <summary>
        /// 返回每一行/列 sum
        /// </summary>
        /// <param name="axis">0 为 列， 1为行</param>
        /// <returns>返回一个二维数组，但只有一行或一列</returns>
        public Tensor Sum(int axis)
        {
            return new Tensor(this.deviceData.Sum(axis));
        }
        public Tensor Mean(int axis)
        {
            return new Tensor(this.deviceData.Mean(axis));
        }
        public Tensor Variance(int axis)
        {
            return new Tensor(this.deviceData.Variance(axis));
        }
        /// <summary>
        /// 按第一列遍历
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public float this[int index]
        {
            get
            {
                return this.deviceData[index];
            }
            set
            {
                this.deviceData[index] = value;
            }
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

        #region static method
        /// <summary>
        /// result = left * right
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        public static void Multiply(Tensor left, Tensor right, Tensor result)
        {
            result.deviceData.GetType().GetMethod("Multiply", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left.deviceData, right.deviceData, result.deviceData });
        }
        /// <summary>
        /// result = left + right
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        public static void Add(Tensor left, Tensor right, Tensor result)
        {
            result.deviceData.GetType().GetMethod("Add", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left.deviceData, right.deviceData, result.deviceData });
        }
        /// <summary>
        /// result = left + right
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        public static void Add(Tensor left, float right, Tensor result)
        {
            result.deviceData.GetType().GetMethod("AddNumber", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left.deviceData, right, result.deviceData });
        }
        /// <summary>
        /// result = left * right
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        public static void Multiply(float left, Tensor right, Tensor result)
        {
            result.deviceData.GetType().GetMethod("MultiplyNumber", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left, right.deviceData, result.deviceData });
        }
        /// <summary>
        /// result = left - right
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        public static void Minus(Tensor left, Tensor right, Tensor result)
        {
            result.deviceData.GetType().GetMethod("Minus", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left.deviceData, right.deviceData, result.deviceData });
        }
        /// <summary>
        /// 矩阵点乘
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="result"></param>
        public static void DotMultiply(Tensor left, Tensor right, Tensor result)
        {
            left.deviceData.GetType().GetMethod("DotMultiply", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left.deviceData, right.deviceData, result.deviceData });
        }
        public static void DotDivide(Tensor left, float right, Tensor result)
        {
            left.deviceData.GetType().GetMethod("DotDivideNumber", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left.deviceData, right, result.deviceData });
        }
        public static void DotDivide(Tensor left, Tensor right, Tensor result)
        {
            left.deviceData.GetType().GetMethod("DotDivide", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left.deviceData, right.deviceData, result.deviceData });
        }
        /// <summary>
        /// 开方
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static void Sqrt(Tensor @in)
        {
            @in.deviceData.GetType().GetMethod("Sqrt", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { @in.deviceData });
        }
        /// <summary>
        /// 矩阵增加一行 Transpose
        /// </summary>
        /// <param name="in">输入矩阵</param>
        /// <param name="value">增加矩阵的值</param>
        /// <param name="result">输出矩阵</param>
        public static void AddLastOneRowWithValue(Tensor @in, float value, Tensor result)
        {
            result.deviceData.GetType().GetMethod("AddOneRowWithValue", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { @in.deviceData, result.deviceData, value, @in.Width });
        }
        public static void Transpose(Tensor @in)
        {
            @in.deviceData.GetType().GetMethod("TransposeSelf", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { @in.deviceData });
        }
        public static void Copy(Tensor @in, Tensor result)
        {
            @in.deviceData.GetType().GetMethod("Copy", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { @in.deviceData, result.deviceData });
        }
        public static void RemoveLastOneRow(Tensor @in)
        {
            @in.deviceData.GetType().GetMethod("RemoveLastOneRow", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { @in.deviceData});
        }
        public static void Relu(Tensor left, Tensor right, Tensor result)
        {
            left.deviceData.GetType().GetMethod("ReluGPU", BindingFlags.Public | BindingFlags.Static).Invoke(null, new object[] { left.deviceData, right.deviceData, result.deviceData });
        }
        #endregion

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
        public static Tensor operator /(float rhs, Tensor lhs)
        {
            return new Tensor(rhs / lhs.deviceData);
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
