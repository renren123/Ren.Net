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
                    // outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss} {Level:u3} {Message}{NewLine}{Exception}", theme: AnsiConsoleTheme.Literate)
                
                .CreateLogger();
        }
        static void Main(string[] args)
        {
            TaskScheduler.UnobservedTaskException += TaskScheduler_UnobservedTaskException;

            InitNetLogging();

            Sequential netWork = new Sequential(new List<NetModule>()
            {
                // layer1
                new Linear(1, 5000),
                new ReLU(),
                //// layer2
                new Linear(5000, 5000),
                new ReLU(),
                //new Linear(10000, 10000),
                //new ReLU(),
                //// layer3
                new Linear(5000, 1),
            });

            netWork.Optimizer = new Adam(learningRate: 0.001F) 
            {
                Device = Device.DeviceTpye.CUDA
            };
            
            netWork.Device = Device.DeviceTpye.CUDA;


            // Sequential netWork = Sequential.Load();
            Log.Information("net: \r\n" + netWork.ToString());

            MSELoss loss = new MSELoss();

            int epoch = 100000;

            long startTime = Stopwatch.GetTimestamp();

            for (int i = 0; i < epoch; i++)
            {
                if (i == epoch - 1)
                {
                    int a = 0;
                }
                var (input, label) = GetTorch();

                Tensor output = netWork.Forward(input);
                var sensitive = loss.CaculateLoss(label, output);

                if (i % 2 == 0)
                {
                    Log.Information($"loss: {sensitive.GetItem()}");
                    //Sequential.Save(netWork);
                }

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
        /// 模拟 函数 y = x + 1
        /// </summary>
        /// <returns></returns>
        static (Tensor input, Tensor label) GetTorch()
        {
            // ########################### CPU ###########################
            {
                //int a = new Random().Next(1, 100);

                //Tensor input = new Tensor(new float[,] { { a } });
                //Tensor label = new Tensor(new float[,] { { a + 1 } });
                //return (input, label);
            }
            // ########################### CPU ###########################

            // ########################### CUDA ###########################
            {
                int a = new Random().Next(1, 100);
                int b = new Random().Next(1, 100);

                //Tensor input = new Tensor(new float[,] { { a  } });
                //Tensor label = new Tensor(new float[,] { { a + 1} });
                Tensor input = new Tensor(5002, 5002, a);
                Tensor label = new Tensor(5002, 5002, a + 1);

                input.Width = label.Width = 1;
                input.Height = label.Height = 1;

                return (input, label);
            }
            // ########################### CUDA ###########################
        }
    }
}
