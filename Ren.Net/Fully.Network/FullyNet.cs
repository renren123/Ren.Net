using Ren.Net.Util;
using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Fully.Network
{
    class FullyNet
    {
        private bool isAdamInit = false;
        private Random r;
        public List<float[]> lossValue;
        /// <summary>
        ///  标准的输出值，用于计算误差项
        /// </summary>
        public float[] StandardOutData { set; get; }
        public List<float[]> InputArray { set; get; }
        public List<float[]> OutSensitive { set; get; }
        public int[] NumberOfLayorNeuron { set; get; }
        public List<float[]> outData { set; get; }
        public List<FullyLayer> fullyLayers;

        public FullyNet()
        {
            r = new Random();
        }
        public bool save(string mainName, string kuozhanName)
        {
            return true;
            /*
            try
            {
                string fileName = "file";
                string fullPath = Directory.GetCurrentDirectory();
                if (Directory.Exists(fullPath + "\\" + fileName) == false)
                {
                    Directory.CreateDirectory(fileName);
                }
                fullPath += "\\" + fileName + "\\" + mainName;
                //MyExcelAction excelAction = new MyExcelAction();

                for (int i = 0; i < fullyLayers.Count; i++)
                {
                    //ReduceMemory.SetProcessWorkingSetSize(System.Diagnostics.Process.GetCurrentProcess().Handle.ToInt32(), -1, -1);
                    if (fullyLayers[i].fullyneurns != null)
                    {
                        TxtAction.FullyLayerToText(fullyLayers[i], fullPath + "_Wi_" + i + kuozhanName);
                        //excelAction.TwoDArrayToExcel(listFullyNetworkLayer[i].Wi, fullPath + "\\" + fileName +"\\"+ mainName + "_Wi_" + i+ kuozhanName);
                        TxtAction.FullyLayerRBToText(fullyLayers[i], fullPath + "_RB_" + i + kuozhanName);
                    }

                }
                return true;
            }
            catch
            {
                return false;
            }
            */
        }
        public bool load(string mainName, string kuozhanName, int cengNumber, int firstLayerNumber)
        {
            /*
            try
            {
                string fileName = "file";
                string fullPath = Directory.GetCurrentDirectory();
                if (Directory.Exists(fullPath + "\\" + fileName) == false)
                {
                    return false;
                }
                for (int j = 0; j < cengNumber; j++)
                {
                    if (fullyLayers == null)
                    {
                        fullyLayers = new List<FullyLayer>(cengNumber);
                    }
                    while (fullyLayers.Count< cengNumber)
                    {
                        fullyLayers.Add(new FullyLayer());
                    }
                }
                NumberOfLayorNeuron[0] = firstLayerNumber;
                fullyLayers = new List<FullyLayer>(NumberOfLayorNeuron.Length);
                for (int i = 0; i < NumberOfLayorNeuron.Length; i++)
                {
                    fullyLayers.Add(new FullyLayer());
                }
                for (int i = 0; i < fullyLayers.Count; i++)
                {
                    if (fullyLayers[i].fullyneurns == null)
                    {
                        fullyLayers[i].fullyneurns = new List<Fullyneuron>(NumberOfLayorNeuron[i]);
                        for (int j = 0; j < NumberOfLayorNeuron[i]; j++)
                        {
                            fullyLayers[i].fullyneurns.Add(new Fullyneuron());
                        }
                    }
                }
             

                for (int i = 0; i < fullyLayers.Count; i++)
                {
                    string fileFullyPathName = fullPath + "\\" + fileName + "\\" + mainName;
                    if (File.Exists(fileFullyPathName + "_Wi_" + i + kuozhanName))
                    {
                        TxtAction.TextToFullyLayer(fullyLayers[i], fileFullyPathName + "_Wi_" + i + kuozhanName);
                    }
                    if (File.Exists(fileFullyPathName + "_RB_" + i + kuozhanName))
                    {
                        TxtAction.RBTextToFullyLayer(fullyLayers[i], fileFullyPathName + "_RB_" + i + kuozhanName);
                    }
                }
            }
            catch (Exception ew)
            {
                System.Windows.Forms.MessageBox.Show("位置：FullyNetwork->load():\n\r" + ew.Message);
                return false;
            }*/
            return true;
        }
        public void updateWi()
        {
            //输出节点
            FullyLayer lastFullyLayer = fullyLayers[fullyLayers.Count - 1];
            for (int i = 0; i < lastFullyLayer.fullyneurns.Count; i++)
            {
                lastFullyLayer.fullyneurns[i].backpropagation(lossValue[i]);
            }
            //隐藏节点
            for (int i = fullyLayers.Count - 1; i > 0; i--)
            {
                backward(fullyLayers[i - 1], fullyLayers[i]);
            }
            //输出敏感度
            if (OutSensitive == null)
            {
                OutSensitive = new List<float[]>(AgentClass.Mini_batchsize);
                for (int i = 0; i < AgentClass.Mini_batchsize; i++)
                {
                    OutSensitive.Add(new float[NumberOfLayorNeuron[0]]);
                }
            }
            for (int i = 0; i < AgentClass.Mini_batchsize; i++)
            {
                for (int j = 0; j < fullyLayers[0].fullyneurns.Count; j++)
                {
                    OutSensitive[i][j] = fullyLayers[0].fullyneurns[j].SensitiveValue[i];
                }
            }
            //更新权值
            //初始化ADAM
            if (isAdamInit == false)
            {
                for (int i = 0; i < fullyLayers.Count - 1; i++)
                {
                    for (int j = 0; j < fullyLayers[i].fullyneurns.Count; j++)
                    {
                        Fullyneuron fullyneuron = fullyLayers[i].fullyneurns[j];
                        if (fullyneuron.S == null)
                        {
                            fullyneuron.S = new float[NumberOfLayorNeuron[i + 1]];
                            fullyneuron.V = new float[NumberOfLayorNeuron[i + 1]];
                        }
                    }
                }
                isAdamInit = true;
            }
            //用于保存各个dw
            float[] dwArray = new float[AgentClass.Mini_batchsize];
            //更新权值
            for (int i = 0; i < fullyLayers.Count - 1; i++)
            {
                float dw = 0;
                for (int j = 0; j < fullyLayers[i].fullyneurns.Count; j++)
                {
                    Fullyneuron fullyneuron = fullyLayers[i].fullyneurns[j];
                    for (int k = 0; k < fullyneuron.Wi.Length; k++)
                    {
                        for (int mini = 0; mini < AgentClass.Mini_batchsize; mini++)
                        {
                            dwArray[mini] = fullyneuron.Xout[mini] * fullyLayers[i + 1].fullyneurns[k].SensitiveValue[mini];
                        }
                        dw = ArrayAction.average(dwArray);
                        fullyneuron.AdamUpdateWi(dw, k);
                        //float quanzhi = Adam.GetAdamNumber(fullyneuron.V[k], fullyneuron.S[k], dw,out fullyneuron.V[k],out fullyneuron.S[k]);
                        //fullyneuron.Wi[k] -= AgentClass.Study_rate * quanzhi;
                    }
                }
            }

        }
        public void backward(FullyLayer lastFullyLayer, FullyLayer presentFullyLayer)
        {
            for (int i = 0; i < lastFullyLayer.fullyneurns.Count; i++)
            {
                float[] temp = GetSensitive(presentFullyLayer, lastFullyLayer.fullyneurns[i].Wi);
                lastFullyLayer.fullyneurns[i].backpropagation(temp);
            }
        }
        public float[] GetSensitive(FullyLayer presentFullyLayer, float[] wi)
        {
            float[] sum = new float[AgentClass.Mini_batchsize];
            for (int i = 0; i < wi.Length; i++)
            {
                for (int j = 0; j < sum.Length; j++)
                {
                    sum[j] += wi[i] * presentFullyLayer.fullyneurns[i].SensitiveValue[j];
                }
            }
            return sum;
        }
        /// <summary>
        /// 初始化权值，np.random.randn(n) * sqrt(2.0/n)
        /// </summary>
        /// <param name="sumInput">输入个数</param>
        /// <returns></returns>
        private float W_value_method(int sumInput)
        {
            float y = (float)r.NextDouble();
            float x = (float)r.NextDouble();
            float number = (float)(Math.Cos(2 * Math.PI * x) * Math.Sqrt(-2 * Math.Log(1 - y)));
            number *= (float)Math.Sqrt(2.0 / sumInput);
            return number;
        }

        public void caculate()
        {
            if (InputArray == null || NumberOfLayorNeuron == null)
            {
                Console.WriteLine("Error!位置：\n\rFullyNet->caculate()");
                return;
            }
            //初始化神经元
            if (fullyLayers == null)
            {
                NumberOfLayorNeuron[0] = InputArray[0].Length;
                fullyLayers = new List<FullyLayer>(NumberOfLayorNeuron.Length);
                for (int i = 0; i < NumberOfLayorNeuron.Length; i++)
                {
                    fullyLayers.Add(new FullyLayer());
                }
                for (int i = 0; i < fullyLayers.Count; i++)
                {
                    if (fullyLayers[i].fullyneurns == null)
                    {
                        fullyLayers[i].fullyneurns = new List<Fullyneuron>(NumberOfLayorNeuron[i]);
                        for (int j = 0; j < NumberOfLayorNeuron[i]; j++)
                        {
                            fullyLayers[i].fullyneurns.Add(new Fullyneuron());
                        }
                    }
                }
                //初始化权值
                for (int i = 0; i < fullyLayers.Count - 1; i++)
                {
                    for (int j = 0; j < fullyLayers[i].fullyneurns.Count; j++)
                    {
                        Fullyneuron fullyneuron = fullyLayers[i].fullyneurns[j];
                        if (fullyneuron.Wi == null)
                        {
                            fullyneuron.Wi = new float[NumberOfLayorNeuron[i + 1]];
                            for (int k = 0; k < NumberOfLayorNeuron[i + 1]; k++)
                            {
                                fullyneuron.Wi[k] = W_value_method(NumberOfLayorNeuron[i]);
                                //fullyneuron.BN.Gamma = W_value_method(NumberOfLayorNeuron[i]);
                                //fullyneuron.BN.Bata = W_value_method(NumberOfLayorNeuron[i]);
                            }
                        }
                    }
                }
            }
            //第一层输入
            FullyLayer firstLayer = fullyLayers[0];
            for (int i = 0; i < firstLayer.fullyneurns.Count; i++)
            {
                float[] XiArray = new float[AgentClass.Mini_batchsize];
                for (int j = 0; j < AgentClass.Mini_batchsize; j++)
                {
                    XiArray[j] = InputArray[j][i];
                }
                firstLayer.fullyneurns[i].forward(XiArray);
            }



            //隐藏层
            for (int i = 1; i < fullyLayers.Count; i++)
            {
                forwardcaculate(fullyLayers[i - 1], fullyLayers[i]);
            }
            //输出
            if (outData == null)
            {
                int lastNetNumber = NumberOfLayorNeuron[NumberOfLayorNeuron.Length - 1];
                outData = new List<float[]>(AgentClass.Mini_batchsize);
                for (int i = 0; i < AgentClass.Mini_batchsize; i++)
                {
                    outData.Add(new float[lastNetNumber]);
                }
            }
            FullyLayer lastFullyLayer = fullyLayers[fullyLayers.Count - 1];
            for (int i = 0; i < AgentClass.Mini_batchsize; i++)
            {
                for (int j = 0; j < NumberOfLayorNeuron[NumberOfLayorNeuron.Length - 1]; j++)
                {
                    outData[i][j] = lastFullyLayer.fullyneurns[j].Xout[i];
                }
                //outData[i] = lastFullyLayer.fullyneurns[i].Xout;
            }
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="lastfullyLayer">上一层</param>
        /// <param name="presentfullyLayer">当前层</param>
        private void forwardcaculate(FullyLayer lastfullyLayer, FullyLayer presentfullyLayer)
        {
            for (int i = 0; i < presentfullyLayer.fullyneurns.Count; i++)
            {
                float[] x_in = new float[AgentClass.Mini_batchsize];
                for (int j = 0; j < AgentClass.Mini_batchsize; j++)
                {
                    x_in[j] = GetWi_Xi(lastfullyLayer, i, j);
                }
                presentfullyLayer.fullyneurns[i].forward(x_in);
            }
        }
        private float GetWi_Xi(FullyLayer lastfullyLayer, int fullyneurnIndex, int Xindex)
        {
            float sum = 0;
            for (int i = 0; i < lastfullyLayer.fullyneurns.Count; i++)
            {
                Fullyneuron fullyneuron = lastfullyLayer.fullyneurns[i];
                sum += fullyneuron.Wi[fullyneurnIndex] * fullyneuron.Xout[Xindex];
            }
            return sum;
        }
    }
}
