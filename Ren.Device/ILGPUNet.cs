using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Ren.Device
{
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
        static void MatrixMinusAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            cView[index] = aView[index] - bView[index];
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



        public ILGPUNet()
        {
        }

        public ILGPUNet(float[,] data)
        {
            var m = data.GetLength(0);
            var ka = data.GetLength(1);
            Data = Accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            Data.CopyFromCPU(data);
        }
        public ILGPUNet(int m, int n, int value)
        {
            float[,] data = new float[m, n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    data[i, j] = value;
                }
            }
            Data = Accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
            Data.CopyFromCPU(data);
        }
        private ILGPUNet(MemoryBuffer2D<float, Stride2D.DenseX> data)
        {
            this.Data?.Dispose();
            this.Data = data;
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
            Data = Accelerator.Allocate2DDenseX<float>(new Index2D(data.GetLength(0), data.GetLength(1)));
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

            Data = Accelerator.Allocate2DDenseX<float>(new Index2D(data.GetLength(0), data.GetLength(1)));
            Data.CopyFromCPU(data);
        }
        public object Clone()
        {
            MemoryBuffer2D<float, Stride2D.DenseX> copy = Accelerator.Allocate2DDenseX<float>(new Index2D(this.Row, this.Column));
            copy.CopyFrom(Data);
            return new ILGPUNet(copy);
        }

        public float RowAverage(int index)
        {
            return Data.View.SubView(new Index2D(index, 0), new Index2D(index, Column)).BaseView.GetAsArray().Average();
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
            float[,] result = new float[Row + 1, Column];
            var copy = Data.GetAsArray2D();

            for (int i = 0; i < Row; i++)
            {
                for (int j = 0; j < Column; j++)
                {
                    result[i, j] = copy[i, j];
                }
            }
            for (int i = 0; i < Column; i++)
            {
                result[Row, i] = value;
            }
            return new ILGPUNet(result);
        }

        public DataInterface RemoveLastOneColumn()
        {
            throw new NotImplementedException();
        }

        public DataInterface RemoveLastOneRow()
        {
            //MemoryBuffer2D<float, Stride2D.DenseX> cBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row - 1, Column));
            //cBuffer.CopyFrom(Data);
            //return new ILGPUNet(cBuffer);

            float[,] removeData = new float[Row - 1, Column];
            var oldData = Data.GetAsArray2D();

            for (int i = 0; i < Row - 1; i++)
            {
                for (int j = 0; j < Column; j++)
                {
                    removeData[i, j] = oldData[i, j];
                }
            }
            return new ILGPUNet(removeData);
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
        /// 这个地方 先 GPU -> CPU -> GPU，目前没有找到合适的转换函数
        /// </summary>
        /// <returns></returns>
        public DataInterface Transpose()
        {
            var result = Data.View.AsTransposed().GetAsArray2D();
            return new ILGPUNet(result);
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

        public DataInterface Relu(DataInterface old)
        {
            var oldData = old as ILGPUNet;

            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(Row, Column));
            ReluKernel(bBuffer.Extent.ToIntIndex(), oldData.Data.View, this.Data.View,  bBuffer.View);
            return new ILGPUNet(bBuffer);
        }

        public float this[int i, int j] { get => Data.View[i, j]; set => Data.View[i, j] = value; }

        public static ILGPUNet operator *(ILGPUNet lhs, ILGPUNet rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> cBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(lhs.Row, rhs.Column));
            MultiplyKernel(cBuffer.Extent.ToIntIndex(), lhs.Data.View, rhs.Data.View, cBuffer.View);
            return new ILGPUNet(cBuffer);
        }
        public static ILGPUNet operator *(float lhs, ILGPUNet rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(rhs.Row, rhs.Column));
            MultiplyNumberKernel(bBuffer.Extent.ToIntIndex(), rhs.Data.View, lhs, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }
        public static ILGPUNet operator *(ILGPUNet lhs, float rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(lhs.Row, lhs.Column));
            MultiplyNumberKernel(bBuffer.Extent.ToIntIndex(), lhs.Data.View, rhs, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }
        public static ILGPUNet operator /(ILGPUNet lhs, float rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(lhs.Row, lhs.Column));
            DivideNumberKernel(bBuffer.Extent.ToIntIndex(), lhs.Data.View, rhs, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }
        public static ILGPUNet operator +(ILGPUNet lhs, ILGPUNet rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> cBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(lhs.Row, rhs.Column));
            AddKernel(cBuffer.Extent.ToIntIndex(), lhs.Data.View, rhs.Data.View, cBuffer.View);
            return new ILGPUNet(cBuffer);
        }
        public static ILGPUNet operator +(ILGPUNet lhs, float rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(lhs.Row, lhs.Column));
            AddNumberKernel(bBuffer.Extent.ToIntIndex(), lhs.Data.View, rhs, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }
        public static ILGPUNet operator -(ILGPUNet lhs, ILGPUNet rhs)
        {
            MemoryBuffer2D<float, Stride2D.DenseX> bBuffer = Accelerator.Allocate2DDenseX<float>(new Index2D(lhs.Row, lhs.Column));
            MinusKernel(bBuffer.Extent.ToIntIndex(), lhs.Data.View, rhs.Data.View, bBuffer.View);
            return new ILGPUNet(bBuffer);
        }
    }
}
