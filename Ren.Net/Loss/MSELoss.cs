using Ren.Net.Objects;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Loss
{
    public class MSELoss
    {
        public Torch CaculateLoss(Torch label, Torch output)
        {
            Torch result = new Torch()
            {
                Data = new List<float[]>(label.Data.Count)
            };

            for (int i = 0; i < output.Data.Count; i++)
            {
                var batchSize = output.Data[i].Length;
                float[] temp = new float[batchSize];

                for (int j = 0; j < batchSize; j++)
                {
                    //temp[j] = label.Data[i][j] - output.Data[i][j];
                    temp[j] = output.Data[i][j] - label.Data[i][j];
                }
                result.Data.Add(temp);
            }
            return result;
        }
    }
}
