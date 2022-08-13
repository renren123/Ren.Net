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
        public int DeviceIndex { get; } = 0;
        public int Row { get => (int)Data.Extent.X; }
        public int Column { get => (int)Data.Extent.Y; }


        public int Width { set; get; }
        public int Height { set; get; }

        public DeviceTpye Device { get; } = DeviceTpye.CUDA;

        private static Context ContextDevice { get; } = Context.CreateDefault();
        // private static Accelerator Accelerator { get; } = ContextDevice.GetCudaDevice(DeviceIndex).CreateAccelerator(ContextDevice);
        private static Accelerator Accelerator { get; } = ContextDevice.GetPreferredDevices(true, true).First().CreateAccelerator(ContextDevice);

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
        /// 矩阵相乘，入参 a * b = c, int 为 a height
        /// </summary>
        private static Action<Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            int,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> ILGPUMultiplyKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                int,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                ILGPUMultiplyAcceleratedKernel);



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

        private static Action<
            Index2D,
            int,
            int,
            ArrayView2D<float, Stride2D.DenseX>> TransposeSelfKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                int,
                int,
                ArrayView2D<float, Stride2D.DenseX>>(
                TransposeAcceleratedKernel);
        
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
            ArrayView2D<float, Stride2D.DenseX>> SetOneRowValueKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                int,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(
                SetOneRowValueAcceleratedKernel);
        /// <summary>
        /// 索引 set 设置值
        /// </summary>
        private static Action<
            Index2D,
            float,
            ArrayView2D<float, Stride2D.DenseX>> SetArrayIndexKernel = Accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(
                SetArrayIndexAcceleratedKernel);




        static void MatrixMultiplyToSelfAcceleratedKernel(
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

        static void ILGPUMultiplyAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            int aHeight,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;

            for (var i = 0; i < aHeight; i++)
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

        static void SetArrayIndexAcceleratedKernel(
            Index2D index,
            float value,
            ArrayView2D<float, Stride2D.DenseX> aView)
        {

            aView[index] = value;
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
            else
            {
                cView[index] = 0F;
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
        static void TransposeAcceleratedKernel(
            Index2D index,
            int width,
            int height,
            ArrayView2D<float, Stride2D.DenseX> aView)
        {
            var x = index.X;
            var y = index.Y;
            if (width >= height && x > y || height >= width && y > x)
            {
                var temp = aView[new Index2D(x, y)];
                aView[new Index2D(x, y)] = aView[new Index2D(y, x)];
                aView[new Index2D(y, x)] = temp;
            }
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
        static void SetOneRowValueAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            int rowIndex,
            float value,
            ArrayView2D<float, Stride2D.DenseX> bView)
        {
            var x = index.X;
            var y = index.Y;

            if (x == rowIndex)
            {
                bView[index] = value;
            }
            else
            {
                bView[index] = aView[index];
            }
            //if (x < rowIndex)
            //{
            //    bView[index] = aView[index];
            //}
            //else if (x > rowIndex)// 所有行往后移动一行
            //{
            //    bView[new Index2D(x + 1, y)] = aView[new Index2D(x, y)];
            //}
            //else // 特定行进行赋值
            //{
            //    bView[new Index2D(x, y)] = value;
            //}
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
        public ILGPUNet(int m, int n, float[,] data)
        {
            Data = Accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
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
            var copyILGPU = new ILGPUNet(copy);
            copyILGPU.Width = this.Width;
            copyILGPU.Height = this.Height;

            return copyILGPU;
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
            SetOneRowValueKernel(bBuffer.Extent.ToIntIndex(), Data.View, Row, value, bBuffer.View);
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
            var array = this.ToArray();
            float sum = 0F;
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    sum += array[i, j];
                }
            }
            return sum / (array.GetLength(0) * array.GetLength(1));


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
        public DataInterface Divide(float rhs, bool divisor = true)
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
        /// <summary>
        /// 保存在 自身中
        /// </summary>
        /// <param name="rhs"></param>
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


        
        public float[,] ToArray()
        {
            float[,] result = new float[this.Width, this.Height];

            var dataArray = this.Data.GetAsArray2D();

            for (int i = 0; i < this.Width; i++)
            {
                for (int j = 0; j < this.Height; j++)
                {
                    result[i, j] = dataArray[i, j];
                }
            }
            return result;
        }


        public void Dispose()
        {
            this.Data.Dispose();
            this.Data = null;
        }
        // public float this[int i, int j] { get => Data.View[i, j]; set => Data.View[i, j] = value; }

        public override string ToString()
        {
            return ToArray().ToString();
        }

        #region static method
        public static void Multiply(DataInterface lhs, DataInterface rhs, DataInterface result)
        {
            ILGPUNet left = lhs as ILGPUNet;
            ILGPUNet right = rhs as ILGPUNet;
            ILGPUNet ret = result as ILGPUNet;
            if (left.Height != right.Width)
            {
                throw new Exception($"ILGPUNet::Multiply [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Width = left.Width;
            ret.Height = right.Height;
            ILGPUMultiplyKernel(new Index2D(ret.Width, ret.Height), left.Data.View, left.Height, right.Data.View, ret.Data.View);
        }
        public static void MultiplyNumber(float lhs, DataInterface rhs, DataInterface result)
        {
            ILGPUNet right = rhs as ILGPUNet;
            ILGPUNet ret = result as ILGPUNet;

            ret.Width = right.Width;
            ret.Height = right.Height;
            MultiplyNumberKernel(new Index2D(ret.Width, ret.Height), right.Data.View, lhs, ret.Data.View);
        }
        public static void Minus(DataInterface lhs, DataInterface rhs, DataInterface result)
        {
            ILGPUNet left = lhs as ILGPUNet;
            ILGPUNet right = rhs as ILGPUNet;
            ILGPUNet ret = result as ILGPUNet;

            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"ILGPUNet::Minus [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Width = left.Width;
            ret.Height = left.Height;

            MinusKernel(new Index2D(ret.Width, ret.Height), left.Data.View, right.Data.View, ret.Data.View);
        }
        public static void Add(DataInterface lhs, DataInterface rhs, DataInterface @out)
        {
            var left = lhs as ILGPUNet;
            var right = rhs as ILGPUNet;
            var result = @out as ILGPUNet;
            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"ILGPUNet::Add [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            result.Width = left.Width;
            result.Height = left.Height;
            AddKernel(new Index2D(result.Width, result.Height), left.Data.View, right.Data.View, result.Data.View);
        }
        public static void AddNumber(DataInterface lhs, float rhs, DataInterface @out)
        {
            var left = lhs as ILGPUNet;
            var result = @out as ILGPUNet;
            result.Width = left.Width;
            result.Height = left.Height;

            AddNumberKernel(new Index2D(result.Width, result.Height), left.Data.View, rhs, result.Data.View);
        }
        public static void DotMultiply(DataInterface lhs, DataInterface rhs, DataInterface result)
        {
            ILGPUNet left = lhs as ILGPUNet;
            ILGPUNet right = rhs as ILGPUNet;
            ILGPUNet ret = result as ILGPUNet;

            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"ILGPUNet::DotMultiply [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Width = left.Width;
            ret.Height = left.Height;

            DotMultiplyKernel(new Index2D(ret.Width, ret.Height), left.Data.View, right.Data.View, ret.Data.View);
        }
        public static void DotDivide(DataInterface lhs, DataInterface rhs, DataInterface result)
        {
            ILGPUNet left = lhs as ILGPUNet;
            ILGPUNet right = rhs as ILGPUNet;
            ILGPUNet ret = result as ILGPUNet;

            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"ILGPUNet::DotDivide [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Width = left.Width;
            ret.Height = left.Height;
            DotDivideKernel(new Index2D(ret.Width, ret.Height), left.Data.View, right.Data.View, ret.Data.View);
        }
        public static void DotDivideNumber(DataInterface lhs, float rhs, DataInterface result)
        {
            ILGPUNet left = lhs as ILGPUNet;
            ILGPUNet ret = result as ILGPUNet;

            ret.Width = left.Width;
            ret.Height = left.Height;
            DivideNumberKernel(new Index2D(ret.Width, ret.Height), left.Data.View, rhs, ret.Data.View);
        }
        public static void Sqrt(DataInterface @in)
        {
            ILGPUNet left = @in as ILGPUNet;
            SqrtKernel(new Index2D(left.Width, left.Height), left.Data.View, left.Data.View);
        }
        public static void AddOneRowWithValue(DataInterface @in, DataInterface result, float value, int row)
        {
            ILGPUNet left = @in as ILGPUNet;
            ILGPUNet right = result as ILGPUNet;
            right.Width = left.Width + 1;
            right.Height = left.Height;
            SetOneRowValueKernel(new Index2D(right.Width, right.Height), left.Data.View, row, value, right.Data.View);
        }
        public static void TransposeSelf(DataInterface @in)
        {
            ILGPUNet left = @in as ILGPUNet;
            int temp = left.Width;
            left.Width = left.Height;
            left.Height = temp;
            TransposeSelfKernel(new Index2D(left.Width, left.Height), left.Width, left.Height, left.Data.View);
        }
        public static void Copy(DataInterface @in, DataInterface result)
        {
            ILGPUNet left = @in as ILGPUNet;
            ILGPUNet right = result as ILGPUNet;
            right.Width = left.Width;
            right.Height = left.Height;
            CopyKernel(new Index2D(left.Width, left.Height), left.Data.View, right.Data.View);
        }
        public static void RemoveLastOneRow(DataInterface @in)
        {
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
            ILGPUNet left = old as ILGPUNet;
            ILGPUNet right = @new as ILGPUNet;
            ILGPUNet ret = result as ILGPUNet;

            if (left.Width != right.Width || left.Height != right.Height)
            {
                throw new Exception($"ILGPUNet::ReluGPU [{left.Width}, {left.Height}] != [{right.Width}, {right.Height}]");
            }
            ret.Width = left.Width;
            ret.Height = left.Height;

            ReluKernel(new Index2D(ret.Width, ret.Height), left.Data.View, right.Data.View, ret.Data.View);
        }

        public DataInterface Relu(DataInterface old)
        {
            throw new NotImplementedException();
        }
        public DataInterface Sum(AxisType axis)
        {
            throw new NotImplementedException();
        }

        public DataInterface Mean(int axis)
        {
            throw new NotImplementedException();
        }

        public DataInterface Variance(int axis)
        {
            throw new NotImplementedException();
        }
        #endregion
    }
}
