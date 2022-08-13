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

namespace Ren.Net.Test
{
    public class Program
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
                new Linear(1, 5000),
                // new BatchNorm1D(1),
                new ReLU(),
                //// layer2
                //new Linear(2, 2),
                // new BatchNorm1D(2),
                //new ReLU(),
                //new Linear(10000, 10000),
                //new ReLU(),
                //// layer3 
                new Linear(5000, 1)
            });

            netWork.Optimizer = new Adam(learningRate: 0.00001F);
            netWork.Device = Device.DeviceTpye.CUDA;
            netWork.Loss = new MSELoss();

            Log.Information("net: \r\n" + netWork.ToString());
            long startTime = Stopwatch.GetTimestamp();
            long spendTime = Stopwatch.GetTimestamp();

            for (int i = 1; i < epoch; i++)
            {
                if (i == epoch - 1)
                {
                    int a = 0;
                }
                var (input, label) = GetTorch();

                Tensor output = netWork.Forward(input);
                Tensor sensitive = netWork.Loss.CaculateLoss(label, output);

                //if (i % 100 == 0)
                //{
                //    Log.Information($"loss: {sensitive.GetItem()}");
                //    // Sequential.Save(netWork, fileName);
                //}
                var timeLast = (Stopwatch.GetTimestamp() - spendTime) * 1000.0 / Stopwatch.Frequency;

                lossQueue.Enqueue(sensitive.GetItem());
                while (lossQueue.Count > 10000)
                {
                    lossQueue.Dequeue();
                }

                if (timeLast > 1000)
                {
                    // Log.Information($"loss: {sensitive.GetItem()}");
                    Log.Information($"epoch: {i}, loss: {lossQueue.Sum() / lossQueue.Count}");
                    spendTime = Stopwatch.GetTimestamp();
                }

                netWork.Backward(sensitive);

                netWork.OptimizerStep();
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
                int length = 50;
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
        static float R()
        {
            return random.Next(1, 100);
        }
    }
}
