using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Ren.Device
{
    /// <summary>
    /// index.X;  index.Y; 意思是当前数组中的一个点，aView.IntExtent.Y 是数组的大小
    /// </summary>
    [Serializable]
    public class ILGPUNet : DataInterface
    {
        public const int DeviceIndex = 0;
        public int Row { get => (int)Data.Extent.X; }
        public int Column { get => (int)Data.Extent.Y; }
        public DeviceTpye Device { get; } = DeviceTpye.CUDA;

        private static Context ContextDevice { get; } = Context.CreateDefault();
        private static Accelerator Accelerator { get; } = ContextDevice.GetCudaDevice(DeviceIndex).CreateAccelerator(ContextDevice);

        [NonSerialized]
        private MemoryBuffer2D<float, Stride2D.DenseX> Data;

        /// <summary>
        /// 矩阵相乘
        /// </summary>
        private static Action<Index2D, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>> MultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMultiplyAcceleratedKernel);
        /// <summary>
        /// 矩阵点乘
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> DotMultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixDotMultiplyAcceleratedKernel);
        /// <summary>
        /// 矩阵点除
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> DotDivideKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixDotDivideAcceleratedKernel);
        /// <summary>
        /// 矩阵相加
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> AddKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixADDAcceleratedKernel);
        /// <summary>
        /// 矩阵相加,结果保存在 B 中
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> ADDToAKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixADDToAAcceleratedKernel);
        /// <summary>
        /// 矩阵相减
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> MinusKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMinusAcceleratedKernel);
        /// <summary>
        /// 矩阵相减,结果赋值给 A 
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> MinusToAKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMinusToAAcceleratedKernel);
        /// <summary>
        /// 矩阵与数字相乘
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            float,
            ArrayView2D<float, Stride2D.DenseX>> MultiplyNumberKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMultiplyNumberAcceleratedKernel);
        /// <summary>
        /// 矩阵与数字相除
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            float,
            ArrayView2D<float, Stride2D.DenseX>> DivideNumberKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixDivideNumberAcceleratedKernel);
        /// <summary>
        /// 矩阵与数字相加
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            float,
            ArrayView2D<float, Stride2D.DenseX>> AddNumberKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixAddNumberAcceleratedKernel);
        
        /// <summary>
        /// 矩阵开平方
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> SqrtKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixSqrtAcceleratedKernel);
        /// <summary>
        /// Relu 计算
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> ReluKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixReluAcceleratedKernel);
        /// <summary>
        /// 拷贝数据
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> CopyKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixCopyAcceleratedKernel);
        /// <summary>
        /// 矩阵转置
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> TransposeKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixTransposeAcceleratedKernel);
        /// <summary>
        /// 移除一行
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            int,
            ArrayView2D<float, Stride2D.DenseX>> RemoveOneRowKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                int,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixRemoveOneRowAcceleratedKernel);
        /// <summary>
        /// 增加一行
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            int,
            float,
            ArrayView2D<float, Stride2D.DenseX>> AddOneRowKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                int,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixAddOneRowAcceleratedKernel);

        


        static void MatrixMultiplyAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;

            for (var i = 0; i < aView.IntExtent.Y; i++)
                sum += aView[new Index2D(x, i)] * bView[new Index2D(i, y)];

            cView[index] = sum;
        }
        static void MatrixDotMultiplyAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            cView[index] = aView[index] * bView[index];
        }
        static void MatrixDotDivideAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            cView[index] = aView[index] / bView[index];
        }
        static void MatrixADDAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            cView[index] = aView[index] + bView[index];
        }
        static void MatrixADDToAAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView)
        {
            aView[index] = aView[index] + bView[index];
        }
        static void MatrixMinusAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            cView[index] = aView[index] - bView[index];
        }
        static void MatrixMinusToAAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView)
        {
            aView[index] = aView[index] - bView[index];
        }
        static void MatrixMultiplyNumberAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            float number,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            cView[index] = aView[index] * number;
        }
        static void MatrixDivideNumberAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            float number,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            cView[index] = aView[index] / number;
        }
        static void MatrixAddNumberAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            float number,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            cView[index] = aView[index] + number;
        }

        static void MatrixSqrtAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView)
        {
           
           bView[index] = (float)Math.Sqrt(aView[index]);
        }
        #region Data Function，功能操作
        static void MatrixReluAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            if (aView[index] > 0F)
            {
                cView[index] = bView[index];
            }
        }

        static void MatrixCopyAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView)
        {
            bView[index] = aView[index];
        }
        static void MatrixTransposeAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView)
        {
            var x = index.X;
            var y = index.Y;

            bView[new Index2D(x, y)] = aView[new Index2D(y, x)];
        }
        static void MatrixRemoveOneRowAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            int rowIndex,
            ArrayView2D<float, Stride2D.DenseX> bView)
        {
            var x = index.X;
            var y = index.Y;
            if (x < rowIndex)
            {
                bView[index] = aView[index];
            }
            else if (x > rowIndex)// 所有行往前移动一行
            {
                bView[new Index2D(x - 1, y)] = aView[new Index2D(x - 1, y)];
            }
        }
        static void MatrixAddOneRowAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            int rowIndex,
            float value,
            ArrayView2D<float, Stride2D.DenseX> bView)
        {
            var x = index.X;
            var y = index.Y;
            if (x < rowIndex)
            {
                bView[index] = aView[index];
            }
            else if (x > rowIndex)// 所有行往后移动一行
            {
                bView[new Index2D(x - 1, y)] = aView[new Index2D(x - 1, y)];
            }
            else // 特定行进行赋值
            {
                bView[new Index2D(x, y)] = value;
            }
        }
        #endregion




        private ILGPUNet(MemoryBuffer2D<float, Stride2D.DenseX> data)
        {
            this.Data?.Dispose();
            this.Data = data;
        }
        public ILGPUNet(float[,] data)
        {
            var m = data.GetLength(0);
            var ka = data.GetLength(1);

            Data = Accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            Data.CopyFromCPU(data);
        }
        public ILGPUNet(int m, int n, float value = 0F)
        {
            float[,] data = new float[m, n];
            if (value != 0F)
            {
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        data[i, j] = value;
                    }
                }
            }
            Data = Accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
            Data.CopyFromCPU(data);
        }
        public ILGPUNet(int m, int n, Func<int, int, float> init)
        {
            float[,] data = new float[m, n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    data[i, j] = init(i, j);
                }
            }
            Data = Accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
            Data.CopyFromCPU(data);
        }
        public object Clone()
        {
            MemoryBuffer2D<float, Stride2D.DenseX> copy = Accelerator.Allocate2DDenseX<float>(new Index2D(this.Row, this.Column));
            copy.CopyFrom(Data);
            return new ILGPUNet(copy);
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public float RowAverage(int index)
        {
            throw new NotImplementedException();
            // return Data.View.SubView(new Index2D(index, 0), new Index2D(index, Column)).BaseView.GetAsArray().Average();
        }

        public float ColumnAverage(int index)
        {
            throw new NotImplementedException();
        }

        public float RowVariance(int index)
        {
            throw new NotImplementedException();
        }

        public float ColumnVariance(int index)
        {
            throw new NotImplementedException();
        }

        public DataInterface AddOneColumnWithValue(int length, float value)
        {
            throw new NotImplementedException();
        }

        public DataInterface AddOneRowWithValue(int length, float value)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row + 1, Column));
            AddOneRowKernel(bBuffer.Extent.ToIntIndex(), Data.View, Row, value, bBuffer.View);
            //var old = Data.GetAsArray2D();
            //var @new = bBuffer.GetAsArray2D();
            return new ILGPUNet(bBuffer);
        }

        public DataInterface RemoveLastOneColumn()
        {
            throw new NotImplementedException();
        }

        public DataInterface RemoveLastOneRow()
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row - 1, Column));
            RemoveOneRowKernel(bBuffer.Extent.ToIntIndex(), Data.View, Row - 1, bBuffer.View);

            //var old = Data.GetAsArray2D();
            //var @new = bBuffer.GetAsArray2D();
            return new ILGPUNet(bBuffer);
        }

        //public void AddColumn(float[] column)
        //{
        //    throw new NotImplementedException();
        //}

        //public void AddRow(float[] column)
        //{
        //    throw new NotImplementedException();
        //}

        //public void InsertColumn(int columnIndex, float[] column)
        //{
        //    throw new NotImplementedException();
        //}

        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <returns></returns>
        public DataInterface Transpose()
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Column, Row));
            TransposeKernel(bBuffer.Extent.ToIntIndex(), Data.View, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }
        public float GetItem()
        {
            return Data.View.BaseView.GetAsArray().Average();
        }

        public DataInterface Sqrt()
        {
            MemoryBuffer2D<float, Stride2D.DenseX> cBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row, Column));
            SqrtKernel(cBuffer.Extent.ToIntIndex(), this.Data.View, cBuffer.View);
            return new ILGPUNet(cBuffer);
        }
        public DataInterface Multiply(DataInterface rhs)
        {
            ILGPUNet right = rhs as ILGPUNet;
            MemoryBuffer2D<float, Stride2D.DenseX> cBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(this.Row, rhs.Column));
            MultiplyKernel(cBuffer.Extent.ToIntIndex(), this.Data.View, right.Data.View, cBuffer.View);
            return new ILGPUNet(cBuffer);
        }
        public DataInterface Multiply(float rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row, Column));
            MultiplyNumberKernel(bBuffer.Extent.ToIntIndex(), Data.View, rhs, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }
        public DataInterface DotMultiply(DataInterface rhs)
        {
            ILGPUNet right = rhs as ILGPUNet;
            MemoryBuffer2D<float, Stride2D.DenseX> cBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(this.Row, rhs.Column));
            DotMultiplyKernel(cBuffer.Extent.ToIntIndex(), this.Data.View, right.Data.View, cBuffer.View);
            return new ILGPUNet(cBuffer);
        }
        public DataInterface DotDivide(DataInterface divisor)
        {
            ILGPUNet right = divisor as ILGPUNet;
            MemoryBuffer2D<float, Stride2D.DenseX> cBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(this.Row, right.Column));
            DotDivideKernel(cBuffer.Extent.ToIntIndex(), this.Data.View, right.Data.View, cBuffer.View);
            return new ILGPUNet(cBuffer);
        }
        public DataInterface Divide(float rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row, Column));
            DivideNumberKernel(bBuffer.Extent.ToIntIndex(), Data.View, rhs, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }

        public DataInterface Add(DataInterface rhs)
        {
            var right = rhs as ILGPUNet;
            if (right.Row != Row || right.Column != Column)
            {
                throw new Exception($"ILGPUNet::Add [{right.Row}, {right.Column}] != [{Row}, {Column}]");
            }
            MemoryBuffer2D<float, Stride2D.DenseX> cBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row, Column));
            AddKernel(cBuffer.Extent.ToIntIndex(), Data.View, right.Data.View, cBuffer.View);
            return new ILGPUNet(cBuffer);
        }

        public void AddToA(DataInterface rhs)
        {
            var right = rhs as ILGPUNet;
            if (right.Row != Row || right.Column != Column)
            {
                throw new Exception($"ILGPUNet::Add [{right.Row}, {right.Column}] != [{Row}, {Column}]");
            }
            ADDToAKernel(right.Data.Extent.ToIntIndex(), right.Data.View, Data.View);
        }

        public DataInterface Add(float rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row, Column));
            AddNumberKernel(bBuffer.Extent.ToIntIndex(), Data.View, rhs, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }

        public DataInterface Minus(DataInterface rhs)
        {
            var right = rhs as ILGPUNet;
            if (right.Row != Row || right.Column != Column)
            {
                throw new Exception($"ILGPUNet::Minus [{right.Row}, {right.Column}] != [{Row}, {Column}]");
            }
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row, Column));
            MinusKernel(bBuffer.Extent.ToIntIndex(), this.Data.View, right.Data.View, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }
        public void MinusToA(DataInterface rhs)
        {
            var right = rhs as ILGPUNet;
            if (right.Row != Row || right.Column != Column)
            {
                throw new Exception($"ILGPUNet::MinusToA [{right.Row}, {right.Column}] != [{Row}, {Column}]");
            }
            MinusToAKernel(right.Data.Extent.ToIntIndex(), this.Data.View, right.Data.View);
        }


        public DataInterface Relu(DataInterface old)
        {
            var oldData = old as ILGPUNet;

            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row, Column));
            ReluKernel(bBuffer.Extent.ToIntIndex(), oldData.Data.View, this.Data.View,  bBuffer.View);
            return new ILGPUNet(bBuffer);
        }

        public void Dispose()
        {
            this.Data.Dispose();
            this.Data = null;
        }
        public float this[int i, int j] { get => Data.View[i, j]; set => Data.View[i, j] = value; }

    }
}
