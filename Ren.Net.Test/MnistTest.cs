using Ren.Net.Objects;
using System.Collections.Generic;
using System;
using Ren.Net.Networks;
using Ren.Net.ActivationFunction;
using Ren.Net.Loss;
using Ren.Net.Optimizers;
using System.Threading.Tasks;
using System.IO;
using System.Reflection;
using Serilog;
using Serilog.Events;
using Serilog.Sinks.SystemConsole.Themes;
using System.Diagnostics;
using Ren.Data;
using System.Drawing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Concurrent;
using System.Linq;

namespace Ren.Net.Test
{
    public class MnistTest
    {
        private static void InitNetLogging()
        {
            var executingDir = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
            var logPath = Path.Combine(executingDir, "log", "net.log");

            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Verbose()
                .MinimumLevel.Override("Microsoft", LogEventLevel.Debug)
                .MinimumLevel.Override("Microsoft.AspNetCore", LogEventLevel.Debug)
                .Enrich.WithThreadId()
                .WriteTo.Async(a => a.File(logPath,
                    LogEventLevel.Verbose,
                    outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff} [{Level:u3}] [{ThreadId:00}] {Message:lj}{NewLine}{Exception}",
                    fileSizeLimitBytes: 31457280,
                    rollOnFileSizeLimit: true,
                    rollingInterval: RollingInterval.Day,
                    shared: true,
                    retainedFileCountLimit: 30))
                .WriteTo.Console(LogEventLevel.Information,
                    outputTemplate: "{Timestamp:HH:mm:ss} {Level:u3} {Message}{NewLine}{Exception}", theme: AnsiConsoleTheme.Literate)

                // outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss} {Level:u3} {Message}{NewLine}{Exception}", theme: AnsiConsoleTheme.Literate)

                .CreateLogger();
        }
        static void Main(string[] args)
        {
            TaskScheduler.UnobservedTaskException += TaskScheduler_UnobservedTaskException;
            InitNetLogging();

            MnistData mnistData = new MnistData(batchSize:200, shuffle: true);
            mnistData.Init();

            string fileName = "file.name";

            Sequential netWork = new Sequential(new List<NetModule>()
            {
                // layer1
                new Linear(784, 100),
                new BatchNorm1D(100),
                new ReLU(),
                //// layer2
                new Linear(100, 100),
                new BatchNorm1D(100),
                new ReLU(),
                //new Linear(10000, 10000),
                //new ReLU(),
                //// layer3  
                new Linear(100, 1)
            });

            netWork.Optimizer = new Adam(learningRate: 0.01F);
            netWork.Device = Device.DeviceTpye.CPU;
            netWork.Loss = new MSELoss();

            //Sequential netWork = Sequential.Load();
            //netWork.Device = Device.DeviceTpye.CPU;

            Log.Information("net: \r\n" + netWork.ToString());
            long startTime = Stopwatch.GetTimestamp();
            long spendTime = Stopwatch.GetTimestamp();

            float loss = 0F;
            int count = 1;

            for (int i = 0; i < 200; i++)
            {
                
                foreach ((Tensor data, Tensor label) item in mnistData)
                {
                    Tensor output = netWork.Forward(item.data);
                    Tensor sensitive = netWork.Loss.CaculateLoss(item.label, output);

                    //if (i % 100 == 0)
                    //{
                    //    Log.Information($"loss: {sensitive.GetItem()}");
                    //    // Sequential.Save(netWork, fileName);
                    //}
                    //if (count % 200 == 0)
                    //{
                    //    Log.Information($"loss: {sensitive.GetItem()}");
                    //}

                    var timeLast = (Stopwatch.GetTimestamp() - spendTime) * 1000.0 / Stopwatch.Frequency;

                    if (timeLast > 1000)
                    {
                        loss += sensitive.GetItem();
                        Log.Information($"epoch: {i}/200, loss: {loss / count}");
                        spendTime = Stopwatch.GetTimestamp();
                    }

                    netWork.Backup(sensitive);
                    netWork.OptimizerStep();
                    count++;
                }
            }
            

            //for (int i = 0; i < epoch; i++)
            //{
            //    if (i == epoch - 1)
            //    {
            //        int a = 0;
            //    }
            //    var (input, label) = GetTorch();

            //    Tensor output = netWork.Forward(input);
            //    Tensor sensitive = netWork.Loss.CaculateLoss(label, output);

            //    if (i % 100 == 0)
            //    {
            //        Log.Information($"loss: {sensitive.GetItem()}");
            //        // Sequential.Save(netWork, fileName);
            //    }

            //    netWork.Backup(sensitive);

            //    netWork.OptimizerStep();
            //}
            long endTime = Stopwatch.GetTimestamp();

            Log.Information("ms: " + ((endTime - startTime) * 1000.0 / Stopwatch.Frequency));

            Log.Information("END");

            Console.ReadKey();
        }
        private static void TaskScheduler_UnobservedTaskException(object sender, UnobservedTaskExceptionEventArgs e)
        {
            Console.Error.WriteLine($"Fatal error: {e.Exception}");
            Log.Error(e.ToString());
            throw new Exception("UnhandledExceptionEventHandler", e.Exception);
        }
        static int count = 0;
        /// <summary>
        /// 模拟 函数 y = x + 1，行是神经元的个数 列是 batchSize
        /// </summary>
        /// <returns></returns>
        static (Tensor input, Tensor label) GetTorch()
        {
            //{
            //    count++;
            //    if(count % 2 == 0)
            //    {
            //        float[,] input = { { 1, 1 } };
            //        float[,] label = { { 2, 2 } };

            //        return (new Tensor(input), new Tensor(label));
            //    }
            //    else
            //    {
            //        //float[,] input = { { 5, 4, 3, 2, 1 } };
            //        //float[,] label = { { 6, 5, 4, 3, 2 } };
            //        float[,] input = { { 2, 2 } };
            //        float[,] label = { { 3, 3 } };

            //        return (new Tensor(input), new Tensor(label));
            //    }
            //}

            //{
            //    float[,] input = { { 1, 2, 3, 4, 5 } };
            //    float[,] label = { { 2, 3, 4, 5, 6 } };

            //    return (new Tensor(input), new Tensor(label));
            //}

            {
                int length = 200;
                float[,] input = new float[1, length];
                float[,] label = new float[1, length];

                for (int j = 0; j < length; j++)
                {
                    input[0, j] = R();
                    label[0, j] = input[0, j] + 1;
                }

                return (new Tensor(input), new Tensor(label));
            }
        }
        static Random random = new Random(DateTime.UtcNow.Millisecond);
        static int R()
        {
            return random.Next(1, 10000);
        }
    }

    public class MnistData : DataLoader
    {
        public override int Length => this.length;

        private int length = 0;
        private string FilePath { set; get; } = @"E:\Project\Ren.Net\Data\mnist\train";
        private int InputNumber { set; get; } = 0;

        private readonly object datasLock = new object();
        private List<MnistItem> Datas { set; get; } = new List<MnistItem>();

        public MnistData(int batchSize = 1, bool shuffle = false) : base(batchSize, shuffle)
        {

        }
        public override (Tensor data, Tensor label) GetItem(int index)
        {
            float[,] dataMap = new float[InputNumber, this.BatchSize];
            float[,] labelMap = new float[1, this.BatchSize];
            int count = 0;
            while (count < BatchSize)
            {
                for (int i = index * BatchSize; i < (index + 1) * BatchSize; i++)
                {
                    var item = Datas[i % Datas.Count];

                    for (int j = 0; j < InputNumber; j++)
                    {
                        dataMap[j, count] = item.Data[j];
                    }
                    labelMap[0, count] = item.Label;
                }
                count++;
            }
            return (new Tensor(dataMap), new Tensor(labelMap));
        }
        public override void Init()
        {
            Log.Information("mnist init data start");
            string[] files = Directory.GetFiles(FilePath, "*.*", SearchOption.AllDirectories);

            Parallel.ForEach(files, file =>
            {
                // 获取某个目录的父目录
                System.IO.DirectoryInfo parentDir = System.IO.Directory.GetParent(file);
                float[] data = LoadPng(file);
                int label = int.Parse(parentDir.Name);
                lock (datasLock)
                {
                    Datas.Add(new MnistItem()
                    {
                        Label = label,
                        Data = data
                    });
                }
            });
            if (Datas.Count != 0)
            {
                if (Datas.Count % BatchSize == 0)
                {
                    length = Datas.Count / BatchSize;
                }
                else
                {
                    length = Datas.Count / BatchSize + 1;
                }
                InputNumber = Datas.First().Data.Length;
            }
            Log.Information("mnist init data end");
        }
        /// <summary>
        /// SixLabors.ImageSharp，处理图像 
        /// https://www.codenong.com/56e5e46af0c9006702d1/
        /// https://www.cnblogs.com/hellotim/p/14023632.html
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public float[] LoadPng(string path)
        {
            using (Image<Rgba32> img = Image.Load<Rgba32>(path))
            {
                float[] array = new float[img.Width * img.Height];

                for (int i = 0; i < img.Width; i++)
                {
                    for (int j = 0; j < img.Height; j++)
                    {
                        var rgb = img[i, j];
                        array[i * img.Height + j] = rgb.G;
                    }
                }
                return array;
            }
        }
        private class MnistItem
        {
            public int Label { set; get; }
            public float[] Data { set; get; }
        }
    }
}
