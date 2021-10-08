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
                    outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss} {Level:u3} {Message}{NewLine}{Exception}", theme: AnsiConsoleTheme.Literate)
                .CreateLogger();
        }
        static void Main(string[] args)
        {
            TaskScheduler.UnobservedTaskException += TaskScheduler_UnobservedTaskException;

            InitNetLogging();

            Sequential netWork = new Sequential(new List<NetModule>()
            {
                // layer1
                new Linear(1, 2),
                new ReLU(),
                // layer2
                //new Linear(2, 2),
                //new ReLU(),
                //// layer3
                new Linear(2, 1),
            });
            netWork.Optimizer = new Adam(learningRate: 0.01F);

            MSELoss loss = new MSELoss();

            int epoch = 500000;

            for (int i = 0; i < epoch; i++)
            {
                if (i == epoch - 1)
                {
                    int a = 0;
                }
                var (input, label) = GetTorch();

                Torch output = netWork.Forward(input);
                var sensitive = loss.CaculateLoss(label, output);

                if (i % 100 == 0)
                    Console.WriteLine($"aim: {label.Data[0][0]} out: {output.Data[0][0]} loss: {sensitive.Data[0][0]}" );

                netWork.Backup(sensitive);

                netWork.OptimizerStep();
            }

            // Gradient Check 
            // GradientCheck(netWork);

            Console.WriteLine("END");

            Console.ReadKey();
        }

        static void GradientCheck(Sequential netWork)
        {
            float epsilon = 0.0001F;
            netWork.ADDGradient(epsilon);
            var (testInput, testLabel) = GetTorch();
            var addResult = netWork.Forward(testInput);
            netWork.ReduceGradient(epsilon * 2);
            var reduceResult = netWork.Forward(testInput);

            float expected_gradient = (reduceResult.Data[0][0] - addResult.Data[0][0]) / (2 * epsilon);

            float checkResult = Math.Abs(addResult.Data[0][0] - reduceResult.Data[0][0]);

            if (checkResult >= 0.0001F)
            {
                Console.WriteLine($"checkout error {checkResult}");
            }
            else
            {
                Console.WriteLine($"checkout success {checkResult}");
            }
        }

       private static void TaskScheduler_UnobservedTaskException(object sender, UnobservedTaskExceptionEventArgs e)
        {
            Console.Error.WriteLine($"Fatal error: {e.Exception}");
            //MiniDump.TryDump("error.dmp");

            //BasicExecutor.Execute(MainProcer.Stop);
            throw new Exception("UnhandledExceptionEventHandler", e.Exception);
        }

       

        /// <summary>
        /// 模拟 函数 y = x + 1
        /// </summary>
        /// <returns></returns>
        static (Torch input, Torch label) GetTorch()
        {
            int batchSize = 1;
            int x = new Random().Next(1, 100);
            Torch input = new Torch()
            {
                Data = new List<float[]>(batchSize)
            };
            input.Data.Add(new float[] { x });

            Torch label = new Torch()
            {
                Data = new List<float[]>(batchSize)
            };
            label.Data.Add(new float[] { x + 1 });

            return (input, label);
        }
    }
}
