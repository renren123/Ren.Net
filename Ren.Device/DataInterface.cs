using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Device
{
    public interface DataInterface : ICloneable, IDisposable
    {
        public DeviceTpye Device { get; }
        public int Row { get; }
        public int Column { get; }
        public int Width { set; get; }
        public int Height { set; get; }

        public float RowAverage(int index);
        public float ColumnAverage(int index);
        public float RowVariance(int index);
        public float ColumnVariance(int index);
        public DataInterface AddOneColumnWithValue(int length, float value);
        public DataInterface AddOneRowWithValue(int length, float value);
        public DataInterface RemoveLastOneColumn();
        public DataInterface RemoveLastOneRow();
        //public void AddColumn(float[] column);
        //public void AddRow(float[] column);
        //public void InsertColumn(int columnIndex, float[] column);
        public DataInterface Transpose();
        /// <summary>
        /// 获取具体数值（待确定 pytorch 是不是同样的作用），取所有值的平均数
        /// </summary>
        /// <returns></returns>
        public float GetItem();

        #region 运算
        public DataInterface DotMultiply(DataInterface right);
        public DataInterface Sqrt();
        public DataInterface DotDivide(DataInterface divisor);
        /// <summary>
        /// lhs * rhs，参数为 右边那个数
        /// </summary>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public DataInterface Multiply(DataInterface rhs);
        // public void Multiply(DataInterface lhs, DataInterface rhs, DataInterface result);
        public DataInterface Multiply(float rhs);
        public DataInterface Divide(float rhs);
        public DataInterface Add(DataInterface rhs);
        public void AddToA(DataInterface rhs);
        public DataInterface Add(float rhs);
        /// <summary>
        /// 减法
        /// </summary>
        /// <param name="rhs"></param>
        /// <returns></returns>
        public DataInterface Minus(DataInterface rhs);
        public void MinusToA(DataInterface rhs);
        #endregion

        #region Net method
        public DataInterface Relu(DataInterface old);
        #endregion

        public float this[int i, int j]
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

        public static DataInterface operator *(DataInterface lhs, DataInterface rhs)
        {
            return lhs.Multiply(rhs);
        }
        public static DataInterface operator *(float lhs, DataInterface rhs)
        {
            return rhs.Multiply(lhs);
        }
        public static DataInterface operator *(DataInterface lhs, float rhs)
        {
            return lhs.Multiply(rhs);
        }
        public static DataInterface operator /(DataInterface lhs, float rhs)
        {
            return lhs.Divide(rhs);
        }
        public static DataInterface operator +(DataInterface lhs, DataInterface rhs)
        {
            return lhs.Add(rhs);
        }
        public static DataInterface operator +(DataInterface lhs, float rhs)
        {
            return lhs.Add(rhs);
        }
        public static DataInterface operator -(DataInterface lhs, DataInterface rhs)
        {
            return lhs.Minus(rhs);
        }
    }
    public enum DeviceTpye
    {
        Default,
        CPU = 1,
        CUDA = 2
    }
}
