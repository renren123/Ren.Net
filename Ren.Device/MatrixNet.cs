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

        public int Width { set; get; }
        public int Height { set; get; }
        public DeviceTpye Device { get; } = DeviceTpye.CPU;

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
        public DataInterface AddOneRowWithValue(int row, float value)
        {
            Vector<float> vector = VBuild.Dense(Column, value);
            var data = this.Data.InsertRow(row, vector);
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
        public DataInterface DotMultiply(DataInterface rhs)
        {
            MatrixNet matrixNet = rhs as MatrixNet;

            if (this.Row == matrixNet.Row && this.Column == matrixNet.Column)
            {
                return new MatrixNet(Matrix<float>.op_DotMultiply(this.Data, matrixNet.Data));
            }
            // 每一列对应相乘
            if (matrixNet.Row == 1 && matrixNet.Column == this.Column)
            {
                Matrix<float> data = MBuild.Dense(this.Row, this.Column, 0F);

                for (int i = 0; i < this.Row; i++)
                {
                    for (int j = 0; j < this.Column; j++)
                    {
                        data[i, j] = this.Data[i, j] * matrixNet.Data[0, j];
                    }
                }
                return new MatrixNet(data);
            }

            if (matrixNet.Row == this.Row && matrixNet.Column == 1)
            {
                Matrix<float> data = MBuild.Dense(this.Row, this.Column, 0F);

                for (int i = 0; i < this.Row; i++)
                {
                    for (int j = 0; j < this.Column; j++)
                    {
                        data[i, j] = this.Data[i, j] * matrixNet.Data[i, 0];
                    }
                }
                return new MatrixNet(data);
            }
            throw new Exception($"MatrixNet::DotMultiply [{this.Row}, {this.Column}] != [{matrixNet.Row}, {matrixNet.Column}]");

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

            if (this.Row == matrixNet.Row && this.Column == matrixNet.Column)
            {
                return new MatrixNet(Matrix<float>.op_DotDivide(this.Data, matrixNet.Data));
            }

            // 每一列对应相除
            if (matrixNet.Row == 1 && matrixNet.Column == this.Column)
            {
                Matrix<float> data = MBuild.Dense(this.Row, this.Column, 0F);

                for (int i = 0; i < this.Row; i++)
                {
                    for (int j = 0; j < this.Column; j++)
                    {
                        data[i, j] = this.Data[i, j] / matrixNet.Data[0, j];
                    }
                }
                return new MatrixNet(data);
            }

            if (matrixNet.Row == this.Row && matrixNet.Column == 1)
            {
                Matrix<float> data = MBuild.Dense(this.Row, this.Column, 0F);

                for (int i = 0; i < this.Row; i++)
                {
                    for (int j = 0; j < this.Column; j++)
                    {
                        data[i, j] = this.Data[i, j] / matrixNet.Data[i, 0];
                    }
                }
                return new MatrixNet(data);
            }
            throw new Exception($"MatrixNet::DotDivide [{this.Row}, {this.Column}] != [{matrixNet.Row}, {matrixNet.Column}]");
        }

        public DataInterface Multiply(float rhs)
        {
            return new MatrixNet(this.Data * rhs);
        }

        public DataInterface Divide(float rhs, bool divisor = true)
        {
            if (divisor)
            {
                return new MatrixNet(this.Data / rhs);
            }
            else
            {
                return new MatrixNet(rhs / this.Data);
            }
        }

        public DataInterface Add(DataInterface rhs)
        {
            MatrixNet matrixNet = rhs as MatrixNet;

            if (this.Row == matrixNet.Row && this.Column == matrixNet.Column)
            {
                return new MatrixNet(this.Data + matrixNet.Data);
            }

            // 每一列对应相加
            if (matrixNet.Row == 1 && matrixNet.Column == this.Column)
            {
                Matrix<float> data = MBuild.Dense(this.Row, this.Column, 0F);

                for (int i = 0; i < this.Row; i++)
                {
                    for (int j = 0; j < this.Column; j++)
                    {
                        data[i, j] = this.Data[i, j] + matrixNet.Data[0, j];
                    }
                }
                return new MatrixNet(data);
            }

            if (matrixNet.Row == this.Row && matrixNet.Column == 1)
            {
                Matrix<float> data = MBuild.Dense(this.Row, this.Column, 0F);

                for (int i = 0; i < this.Row; i++)
                {
                    for (int j = 0; j < this.Column; j++)
                    {
                        data[i, j] = this.Data[i, j] + matrixNet.Data[i, 0];
                    }
                }
                return new MatrixNet(data);
            }
            throw new Exception($"MatrixNet::Add [{this.Row}, {this.Column}] != [{matrixNet.Row}, {matrixNet.Column}]");
        }

        public DataInterface Add(float rhs)
        {
            return new MatrixNet(this.Data + rhs);
        }

        public DataInterface Minus(DataInterface rhs)
        {
            MatrixNet matrixNet = rhs as MatrixNet;

            if(this.Row == matrixNet.Row && this.Column == matrixNet.Column)
            {
                return new MatrixNet(this.Data - matrixNet.Data);
            }
            // 每一列对应相减
            if (matrixNet.Row == 1 && matrixNet.Column == this.Column)
            {
                Matrix<float> data = MBuild.Dense(this.Row, this.Column, 0F);

                for (int i = 0; i < this.Row; i++)
                {
                    for (int j = 0; j < this.Column; j++)
                    {
                        data[i, j] = this.Data[i, j] - matrixNet.Data[0, j];
                    }
                }
                return new MatrixNet(data);
            }

            if (matrixNet.Row ==  this.Row && matrixNet.Column == 1)
            {
                Matrix<float> data = MBuild.Dense(this.Row, this.Column, 0F);

                for (int i = 0; i < this.Row; i++)
                {
                    for (int j = 0; j < this.Column; j++)
                    {
                        data[i, j] = this.Data[i, j] - matrixNet.Data[i, 0];
                    }
                }
                return new MatrixNet(data);
            }
            throw new Exception($"MatrixNet::Minus [{this.Row}, {this.Column}] != [{matrixNet.Row}, {matrixNet.Column}]");
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


        #region static method
        public static void Multiply(DataInterface lhs, DataInterface rhs, DataInterface result)
        {
            MatrixNet left = lhs as MatrixNet;
            MatrixNet right = rhs as MatrixNet;
            MatrixNet ret = result as MatrixNet;
            if (left.Height != right.Width)
            {
                throw new Exception($"MatrixNet::Multiply [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Data = left.Data * right.Data;
        }
        public static void MultiplyNumber(float lhs, DataInterface rhs, DataInterface result)
        {
            MatrixNet right = rhs as MatrixNet;
            MatrixNet ret = result as MatrixNet;

            ret.Data = lhs * right.Data;
        }
        public static void Minus(DataInterface lhs, DataInterface rhs, DataInterface result)
        {
            MatrixNet left = lhs as MatrixNet;
            MatrixNet right = rhs as MatrixNet;
            MatrixNet ret = result as MatrixNet;

            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"MatrixNet::Minus [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }

            result = lhs.Minus(lhs);
        }
        public static void Add(DataInterface lhs, DataInterface rhs, DataInterface @out)
        {
            var left = lhs as MatrixNet;
            var right = rhs as MatrixNet;
            var result = @out as MatrixNet;
            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"MatrixNet::Add [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }

            @out = lhs + rhs;
        }
        public static void AddNumber(DataInterface lhs, float rhs, DataInterface @out)
        {
            var left = lhs as MatrixNet;
            var result = @out as MatrixNet;
            result.Width = left.Width;
            result.Height = left.Height;

            @out = lhs + rhs;
        }
        public static void DotMultiply(DataInterface lhs, DataInterface rhs, DataInterface result)
        {
            MatrixNet left = lhs as MatrixNet;
            MatrixNet right = rhs as MatrixNet;
            MatrixNet ret = result as MatrixNet;

            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"MatrixNet::DotMultiply [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Width = left.Width;
            ret.Height = left.Height;

            result = lhs.DotMultiply(rhs);
        }
        public static void DotDivide(DataInterface lhs, DataInterface rhs, DataInterface result)
        {
            MatrixNet left = lhs as MatrixNet;
            MatrixNet right = rhs as MatrixNet;
            MatrixNet ret = result as MatrixNet;

            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"MatrixNet::DotDivide [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Width = left.Width;
            ret.Height = left.Height;

            result = lhs.DotDivide(rhs);
        }
        public static void DotDivideNumber(DataInterface lhs, float rhs, DataInterface result)
        {
            MatrixNet left = lhs as MatrixNet;
            MatrixNet ret = result as MatrixNet;

            ret.Width = left.Width;
            ret.Height = left.Height;

            result = lhs / rhs;
        }
        public static void Sqrt(DataInterface @in)
        {
            @in = @in.Sqrt();
        }
        public static void AddOneRowWithValue(DataInterface @in, DataInterface result, float value, int row)
        {
            MatrixNet left = @in as MatrixNet;
            MatrixNet right = result as MatrixNet;
            right.Width = left.Width + 1;
            right.Height = left.Height;

            Vector<float> vector = VBuild.Dense(right.Height, value);
            right = new MatrixNet(left.Data.InsertRow(right.Height, vector));
        }
        public static void TransposeSelf(DataInterface @in)
        {
            MatrixNet left = @in as MatrixNet;
            int temp = left.Width;
            left.Width = left.Height;
            left.Height = temp;

            @in = @in.Transpose();
        }
        public static void Copy(DataInterface @in, DataInterface result)
        {
            MatrixNet left = @in as MatrixNet;
            MatrixNet right = result as MatrixNet;
            right.Width = left.Width;
            right.Height = left.Height;

            result = @in.Clone() as DataInterface;
        }
        public static void RemoveLastOneRow(DataInterface @in)
        {
            @in = @in.RemoveLastOneRow();
            @in.Width -= 1;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="old">以前的结果</param>
        /// <param name="new">新的结果</param>
        /// <param name="result">赋值</param>
        public static void ReluGPU(DataInterface old, DataInterface @new, DataInterface result)
        {
            throw new NotImplementedException();

            MatrixNet oldData = old as MatrixNet;
            MatrixNet right = @new as MatrixNet;
            MatrixNet ret = result as MatrixNet;

            if (oldData.Width != right.Width || oldData.Height != right.Height)
            {
                throw new Exception($"MatrixNet::ReluGPU [{oldData.Width}, {oldData.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Width = oldData.Width;
            ret.Height = oldData.Height;

            for (int i = 0; i < oldData.Row; i++)
            {
                for (int j = 0; j < oldData.Column; j++)
                {
                    if (oldData[i, j] < 0F)
                    {
                        ret[i, j] = 0F;
                    }
                    else
                    {
                        ret[i, j] = right[i, j];
                    }
                }
            }
        }

        public float[,] ToArray()
        {
            return this.Data.ToArray();
        }

        public DataInterface Sum(int axis)
        {
            switch (axis)
            {
                case 0:
                    {
                        Matrix<float> data = MBuild.Dense(1, this.Column, 0F);

                        var colSum = this.Data.ColumnSums();

                        for (int i = 0; i < this.Column; i++)
                        {
                            data[0, i] = colSum[i];
                        }
                        return new MatrixNet(data);
                    }
                case 1:
                    {
                        Matrix<float> data = MBuild.Dense(this.Row, 1, 0F);

                        var rowSum = this.Data.RowSums();

                        for (int i = 0; i < this.Row; i++)
                        {
                            data[i, 0] = rowSum[i];
                        }
                        return new MatrixNet(data);
                    }
                default:
                    throw new Exception($"Sum {axis}");
            }
        }

        public DataInterface Mean(int axis)
        {
            switch (axis)
            {
                case 0: // 求每一列 的均值
                    {
                        Matrix<float> data = MBuild.Dense(1, this.Column, 0F);

                        var colSum = this.Data.ColumnSums();

                        for (int i = 0; i < this.Column; i++)
                        {
                            data[0, i] = colSum[i] / this.Row;
                        }
                        return new MatrixNet(data);
                    }
                case 1: // 求每一行 的均值
                    {
                        Matrix<float> data = MBuild.Dense(this.Row, 1, 0F);

                        var rowSum = this.Data.RowSums();

                        for (int i = 0; i < this.Row; i++)
                        {
                            data[i, 0] = rowSum[i] / this.Column;
                        }
                        return new MatrixNet(data);
                    }
                default:
                    throw new Exception($"Mean {axis}");
            }
        }

        public DataInterface Variance(int axis)
        {
            switch (axis)
            {
                case 0: // 求每一列 的均值
                    {
                        Matrix<float> data = MBuild.Dense(1, this.Column, 0F);

                        for (int i = 0; i < this.Column; i++)
                        {
                            float sum = 0F;
                            var col = this.Data.Column(i);
                            float average = col.Average();

                            for (int j = 0; j < this.Row; j++)
                            {
                                sum += (col[j] - average) * (col[j] - average);
                            }
                            data[0, i] = sum / this.Row;
                        }
                        return new MatrixNet(data);
                    }
                case 1: // 求每一行 的均值
                    {
                        Matrix<float> data = MBuild.Dense(this.Row, 1, 0F);

                        for (int i = 0; i < this.Row; i++)
                        {
                            float sum = 0F;
                            var row = this.Data.Row(i);
                            float average = row.Average();

                            for (int j = 0; j < this.Column; j++)
                            {
                                sum += (row[j] - average) * (row[j] - average);
                            }
                            data[i, 0] = sum / this.Column;
                        }
                        return new MatrixNet(data);
                    }
                default:
                    throw new Exception($"Variance {axis}");
            }
        }
        #endregion

    }
}
