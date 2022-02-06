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
using System.Linq;
using Ren.Net.Extensions;
using Ren.Data;

namespace Ren.Net.Test
{
    public class FunctionalRegressionTest
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

            string fileName = "file.name";
            int epoch = 10000000;
            Queue<float> lossQueue = new Queue<float>(10002);

            Sequential netWork = new Sequential(new List<NetModule>()
            {
                // layer1
                new Linear(1, 1),
                // new BatchNorm1D(1),
                new ReLU(),
                //// layer2
                //new Linear(2, 2),
                // new BatchNorm1D(2),
                //new ReLU(),
                //new Linear(10000, 10000),
                //new ReLU(),
                //// layer3 
                new Linear(1, 1)
            });

            MnistData mnistData = new MnistData(batchSize: 10, shuffle: true);
            mnistData.Init();

            netWork.Optimizer = new Adam(learningRate: 0.00001F);
            netWork.Device = Device.DeviceTpye.CPU;
            netWork.Loss = new MSELoss();

            Log.Information("net: \r\n" + netWork.ToString());
            long startTime = Stopwatch.GetTimestamp();
            long spendTime = Stopwatch.GetTimestamp();

            for (int i = 0; i < 20000000; i++)
            {
                foreach ((Tensor data, Tensor label) item in mnistData)
                {
                    Tensor output = netWork.Forward(item.data);
                    Tensor sensitive = netWork.Loss.CaculateLoss(item.label, output);

                    var timeLast = (Stopwatch.GetTimestamp() - spendTime) * 1000.0 / Stopwatch.Frequency;

                    lossQueue.Enqueue(sensitive.GetItem());
                    while (lossQueue.Count > 10000)
                    {
                        lossQueue.Dequeue();
                    }

                    if (timeLast > 1000)
                    {
                        Log.Information($"loss: {lossQueue.Sum() / lossQueue.Count}");
                        spendTime = Stopwatch.GetTimestamp();
                    }

                    netWork.Backup(sensitive);

                    netWork.OptimizerStep();
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

        class MnistData : DataLoader
        {
            public override int Length => length;

            private Random random = new Random(DateTime.UtcNow.Millisecond);
            private int length = 18;

            public MnistData(int batchSize = 1, bool shuffle = false) : base(batchSize, shuffle)
            {

            }
            public override void Init()
            {

            }
            

            public override (Tensor data, Tensor label) GetItem(int index)
            {
                float[,] input = new float[1, 2];
                float[,] label = new float[1, 2];


                input[0, 0] = R();
                label[0, 0] = input[0, 0] + 1;

                input[0, 1] = R();
                label[0, 1] = input[0, 1] + 1;

                return (new Tensor(input), new Tensor(label));
            }

            
            float R()
            {
                return random.Next(1, length);
            }
        }
    }
}
