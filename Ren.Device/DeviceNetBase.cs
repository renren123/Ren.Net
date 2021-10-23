using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Device
{
    public class DeviceNetBase : ICloneable
    {
        public virtual int Column { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public virtual int Row { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public DeviceNetBase () 
        {
            throw new NotImplementedException();
        }
        public DeviceNetBase(float[,] data) 
        {
            throw new NotImplementedException();
        }
        public DeviceNetBase(int m, int n, float value)
        {
            throw new NotImplementedException();
        }
        public DeviceNetBase(int m, int n, Func<int, int, float> init)
        {
            throw new NotImplementedException();
        }
        public virtual object Clone()
        {
            throw new NotImplementedException();
        }
        public virtual float RowAverage(int i)
        {
            throw new NotImplementedException();
        }
        public virtual float ColumnAverage(int j)
        {
            throw new NotImplementedException();
        }
        public virtual float RowVariance(int index)
        {
            throw new NotImplementedException();
        }
        public virtual float ColumnVariance(int index)
        {
            throw new NotImplementedException();
        }
        public virtual void AddOneColumnWithValue(int length, float value)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// 增加一行 赋值为 value
        /// </summary>
        /// <param name="length"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public virtual void AddOneRowWithValue(int length, float value)
        {
            throw new NotImplementedException();
        }
        public virtual void RemoveLastOneColumn()
        {
            throw new NotImplementedException();
        }
        public virtual void RemoveLastOneRow()
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// 增加一列，加到最后
        /// </summary>
        /// <param name="column"></param>
        public virtual void AddColumn(float[] column)
        {
            throw new NotImplementedException();
        }
        public virtual void AddRow(float[] column)
        {
            throw new NotImplementedException();
        }
        public virtual void InsertColumn(int columnIndex, float[] column)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <returns></returns>
        public virtual void Transpose()
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// 获取具体数值（待确定 pytorch 是不是同样的作用）
        /// </summary>
        /// <returns></returns>
        public virtual float GetItem()
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// 矩阵点乘，对应位相乘
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public virtual void DotMultiply(DeviceNetBase data)
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// 开方
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public virtual void Sqrt()
        {
            throw new NotImplementedException();
        }
        /// <summary>
        /// 点除，对应位相除
        /// </summary>
        /// <param name="dividend"></param>
        /// <param name="divisor"></param>
        /// <returns></returns>
        public virtual void DotDivide(DeviceNetBase divisor)
        {
            throw new NotImplementedException();
        }
        public virtual float this[int i, int j]
        {
            get
            {
                throw new NotImplementedException();
            }
            set
            {
                throw new NotImplementedException();
            }
        }
        public  static DeviceNetBase operator *(DeviceNetBase lhs, DeviceNetBase rhs)
        {
            throw new NotImplementedException();
        }
        public static DeviceNetBase operator *(float lhs, DeviceNetBase rhs)
        {
            throw new NotImplementedException();
        }
        public static DeviceNetBase operator *(DeviceNetBase lhs, float rhs)
        {
            throw new NotImplementedException();
        }
        public static DeviceNetBase operator /(DeviceNetBase lhs, float rhs)
        {
            throw new NotImplementedException();
        }
        public static DeviceNetBase operator +(DeviceNetBase lhs, DeviceNetBase rhs)
        {
            throw new NotImplementedException();
        }
        public static DeviceNetBase operator +(DeviceNetBase lhs, float rhs)
        {
            throw new NotImplementedException();
        }
        public static DeviceNetBase operator -(DeviceNetBase lhs, DeviceNetBase rhs)
        {
            throw new NotImplementedException();
        }
    }
}
