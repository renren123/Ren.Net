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
                .CreateLogger();
        }
        static void Main(string[] args)
        {
            TaskScheduler.UnobservedTaskException += TaskScheduler_UnobservedTaskException;
            InitNetLogging();

            MnistData mnistData = new MnistData(batchSize:60, shuffle: true);
            mnistData.Init();

            string fileName = "file.name";

            Sequential netWork = new Sequential(new List<NetModule>()
            {
                // layer1
                new Linear(784, 10),
                new BatchNorm1D(10),
                new ReLU(),
                // layer2
                new Linear(10, 10),
                new BatchNorm1D(10),
                new ReLU(),
                // layer3  
                new Linear(10, 1)
            });

            netWork.Optimizer = new Adam(learningRate: 0.001F);
            netWork.Device = Device.DeviceTpye.CPU;
            netWork.Loss = new MSELoss();

            //Sequential netWork = Sequential.Load();
            //netWork.Device = Device.DeviceTpye.CPU;

            Log.Information("net: \r\n" + netWork.ToString());
            long startTime = Stopwatch.GetTimestamp();
            long spendTime = Stopwatch.GetTimestamp();

            float loss = 0F;
            int count = 1;

            for (int i = 0; i < 20000; i++)
            {
                foreach ((Tensor data, Tensor label) item in mnistData)
                {
                    Tensor output = netWork.Forward(item.data);
                    Tensor sensitive = netWork.Loss.CaculateLoss(item.label, output);

                    var timeLast = (Stopwatch.GetTimestamp() - spendTime) * 1000.0 / Stopwatch.Frequency;

                    if (timeLast > 1000)
                    {
                        loss += sensitive.GetItem();
                        Log.Information($"epoch: {i}/200, loss: {loss / count}");
                        spendTime = Stopwatch.GetTimestamp();
                    }

                    netWork.Backward(sensitive);
                    netWork.OptimizerStep();
                    count++;
                }
            }
            
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
    }

    class MnistData : DataLoader
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
            float[,] dataMap = new float[InputNumber, 1];
            float[,] labelMap = new float[1, 1];

            for (int i = 0; i < InputNumber; i++)
            {
                dataMap[i, 0] = Datas[index].Data[i];
            }
            labelMap[0, 0] = Datas[index].Label;

            return (new Tensor(dataMap), new Tensor(labelMap));
        }
        public void Init()
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
