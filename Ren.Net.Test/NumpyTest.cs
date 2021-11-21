using Numpy;
using Numpy.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ren.Net.Test
{
    public class NumpyTest
    {
        NDarray running_mean ;
        NDarray running_var ;
        float momentum = 0.99F;
        float eps = 0.001F;
         
        NDarray beta = np.zeros(new Shape(new int[] { 2 }));
        NDarray gamma = np.ones(new Shape(new int[] { 2 }));

        NDarray x_minus_mean;
        NDarray var_plus_eps;

        NDarray x_hat;

        public NumpyTest()
        {
            running_mean = np.zeros(new Shape(new int[] { 2 }));
            running_var = np.zeros(new Shape(new int[] { 2 }));
        }

        public NDarray forward(NDarray x)
        {
            NDarray x_mean = x.mean(axis: 0);
            NDarray x_var = x.var(axis: 0);

            // Console.WriteLine(x_mean);

            running_mean = (1 - momentum) * x_mean + momentum * running_mean;
            running_var = (1 - momentum) * x_var + momentum * running_var;

            x_hat = (x - x_mean) / np.sqrt(x_var + eps);
            NDarray y = gamma * x_hat + beta;

            x_minus_mean = x - x_mean;
            var_plus_eps = x_var + eps;
            return y;
        }
        public NDarray backup(NDarray dout)
        {
            int N = dout.shape.Dimensions[0];

            NDarray dgamma = np.sum(x_hat * dout, axis: 0);
            NDarray dbeta = np.sum(dout, axis: 0);

            var matmulResult = np.matmul(np.ones((N, 1)), gamma.reshape((1, -1)));

            NDarray dx_ = matmulResult * dout;
            NDarray dx = N * dx_ - np.sum(dx_, axis: 0) - x_hat * np.sum(dx_ * x_hat, axis: 0);
            dx *= (1.0 / N) / np.sqrt(var_plus_eps);

            beta -= dbeta;
            gamma -= dgamma;

            return dx;
        }

        static void Main(string[] args)
        {
            var bn = new NumpyTest();

            var a = np.array(new float[3, 2]
            {
                { 1F, 2F },
                { 1F, 3F },
                { 1F, 4F }
            });
            var dout = np.array(new float[3, 2]
            {
                { 2F, 2F },
                { 1F, 1F },
                { 1F, 1F }
            });

            for (int i = 0; i < 2; i++)
            {
                var result = bn.forward(a);
                var dresult = bn.backup(dout);

                Console.WriteLine(result);
                Console.WriteLine();
                Console.WriteLine(dresult);
                Console.WriteLine("#########################");
            }

            //var result = bn.forward(a);
            //var dresult = bn.backup(dout);

            //Console.WriteLine(result);
            //Console.WriteLine();
            //Console.WriteLine(dresult);

            Console.ReadKey();
        }
    }
}
