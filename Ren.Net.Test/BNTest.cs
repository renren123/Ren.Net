using Ren.Net.ActivationFunction;
using Ren.Net.Loss;
using Ren.Net.Networks;
using Ren.Net.Objects;
using Ren.Net.Optimizers;
using Serilog;
using Serilog.Events;
using Serilog.Sinks.SystemConsole.Themes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Ren.Net.Test
{
    class BNTest
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

            string fileName = "file.name";
            int epoch = 3;

            Sequential netWork = new Sequential(new List<NetModule>()
            {
                // layer1
                new BatchNorm1D(2),
            });

            netWork.Optimizer = new Adam(learningRate: 0.01F);
            netWork.Device = Device.DeviceTpye.CPU;
            netWork.Loss = new MSELoss();

            // MSELoss loss = new MSELoss();

            //Sequential netWork = Sequential.Load();
            //netWork.Device = Device.DeviceTpye.CPU;

            Log.Information("net: \r\n" + netWork.ToString());
            long startTime = Stopwatch.GetTimestamp();

            for (int i = 0; i < epoch; i++)
            {
                if (i == epoch - 1)
                {
                    int a = 0;
                }
                var (input, label) = GetTorch();
                Log.Information("count: " + (i + 1));
                Tensor output = netWork.Forward(input);
                Console.WriteLine("output");
                PrintArray(output.ToArray());

                // Tensor sensitive = loss.CaculateLoss(label, output);
                // Tensor sensitive = netWork.Loss.CaculateLoss(input, label);
                Tensor sensitive = netWork.Loss.CaculateLoss(label, output);

                //Console.WriteLine("sensitive");
                //PrintArray(sensitive.ToArray());

                Log.Information($"loss: {sensitive.GetItem()}");

                netWork.Backup(sensitive);

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
        /// <summary>
        /// 模拟 函数 y = x + 1，行是神经元的个数 列是 batchSize
        /// </summary>
        /// <returns></returns>
        static (Tensor input, Tensor label) GetTorch()
        {
            {
                float[,] input =
                {
                    { 1, 1, 1 },
                    { 2, 3, 4 },
                };
                float[,] label =
                {
                    { -2, -1, -1 },
                    { -2, -1, -1 },
                };

                return (new Tensor(input), new Tensor(label));
            }

            {
                int length = 20;
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
            return random.Next(1, 1000);
        }
        static void PrintArray(float[,] array)
        {
            Console.WriteLine();
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    Console.Write($"{array[i, j]} ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }
}
