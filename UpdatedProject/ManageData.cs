using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;
using System.Runtime.InteropServices;
using System.Threading;
using System.Drawing.Text;
using System.Diagnostics.Eventing.Reader;
using System.Runtime.InteropServices.ComTypes;
using System.Drawing;
using Nett;

namespace UpdatedProject
{

    internal class ManageData
    {
        NetInIt netinit = new NetInIt();


        public Matrix<double> weights;
        public Vector<double> BiasVector;


        public Matrix<double> GetWeight(int layer)
        {
            SetWeightDimentions(layer);

            string Filename = $"Data\\layer {layer}\\Weights.txt";
            int k = 0;


            if (File.Exists(Filename))
            {
                lock (Program.filelock)
                {
                    try
                    {
                        while (!Program.isfree)
                        {
                            Monitor.Wait(Program.filelock);
                        }
                        Program.isfree = false;
                        string content = File.ReadAllText(Filename);
                        string[] vals = content.Split(',');
                        for (int i = 0; i < weights.ColumnCount - 1; i++)
                        {
                            for (int j = 0; j < weights.RowCount - 1; j++)
                            {
                                weights[j, i] = Convert.ToDouble(vals[k]);
                                k++;

                            }
                        }
                        return weights;

                    }
                    finally
                    {
                        Program.isfree = true;
                        Monitor.Pulse(Program.filelock);
                    }
                }
            }
            else
            {
                Console.WriteLine("Weights dont exist \n Remaking file...");
                
                netinit.WeightGen(layer);
                GetWeight(layer);
                return weights;

            }
        }

        public Vector<double> getBias(int layer)
        {
            SetBiasDimentions(layer);

            string filename = $"Data\\layer {layer}\\Bias.txt";
            int k = 0;

            if (File.Exists(filename))
            {
                lock (Program.filelock)
                {
                    try
                    {
                        while (!Program.isfree)
                        {
                            Monitor.Wait(Program.filelock);
                        }
                        Program.isfree = false;

                        string content = File.ReadAllText(filename);
                        string[] vals = content.Split(',');
                        for (int i = 0; i < BiasVector.Count; i++)
                        {
                            BiasVector[i] = Convert.ToDouble(vals[k]);
                            k++;
                        }
                        return BiasVector;
                    }
                    finally
                    {
                        Program.isfree = true;
                        Monitor.Pulse(Program.filelock);
                    }
                }
            }
            else
            {
                Console.WriteLine("Bias does not exist \n remaking file...");
                netinit.BiasGen(layer);
                getBias(layer);
                return BiasVector;
            }
        }

        public Matrix<double> GetImage()
        {
            string FileName = "Data\\ImageToProcess";

            if (Directory.Exists(FileName))
            {
                Console.WriteLine("file exists");
                string[] files = Directory.GetFiles(FileName);
                string file = files[0];

                using (Bitmap image = new Bitmap(file))
                {
                    Matrix<double> RGBVal = Matrix<double>.Build.DenseOfArray(new Double[image.Width, image.Height]);

                    for (int y = 0; y < image.Height; y++)
                    {
                        for (int x = 0; x < image.Width; x++)
                        {
                            Color color = image.GetPixel(x, y);

                            double NormColor = color.GetBrightness();

                            RGBVal[x, y] = NormColor;


                        }
                    }
                    return RGBVal;
                }
            }
            else
            {
                Console.WriteLine("File doesnt exist, remaking File...");

                Directory.CreateDirectory(FileName);
                return GetImage();
            }
            
        }


        public Matrix<double> LayerVectorGen(int Pass)
        {
            ImageHandle imageHandle = new ImageHandle();

            string filename = $"image_{Pass}_";

            Matrix<double> LayerVector = imageHandle.NormRGB(filename, Pass);
            return LayerVector;
        }

        public void SaveWeights(Matrix<double> weights, int layer)
        {


            string filename = $"Data\\Layer {layer}\\Weights.txt";

            var weight = new string[weights.ColumnCount * weights.RowCount];

            int k = 0;

            for(int i = 0; i < weights.ColumnCount; i++)
            {
                for(int j = 0; j < weights.RowCount; j++)
                {
                    weight[k] = weights[j, i].ToString();
                    k++;
                }
            }

            lock (Program.filelock)
            {
                try
                {
                    while (!Program.isfree)
                    {
                        Monitor.Wait(Program.filelock);
                    }
                    Program.isfree = false;

                    File.WriteAllText(filename, string.Join(",", weight));

                }
                finally
                {
                    Program.isfree = true;
                    Monitor.Pulse(Program.filelock);
                }
            }
        }
        
        public void SaveBias(Vector<double> Bias, int layer)
        {
            string filename = $"Data\\Layer {layer}\\Bias.txt";

            string[] bias = new string[Bias.Count];

            for(int i = 0;i < Bias.Count; i++)
            {
                bias[i] = Bias[i].ToString();
            }

            lock (Program.filelock)
            {
                try
                {
                    while (!Program.isfree)
                    {
                        Monitor.Wait(Program.filelock);
                    }
                    Program.isfree = false;

                    File.WriteAllText(filename, string.Join(",", bias));

                }
                finally
                {
                    Program.isfree = true;
                    Monitor.Pulse(Program.filelock);
                }
            }
        }
        void SetWeightDimentions(int layer)
        {
            int dimention1;
            int dimention2;

            if (layer > TOMLHandle.LayerCount - 2)
            {
                dimention1 = TOMLHandle.GetHiddenLayerCount()[layer];
                dimention2 = TOMLHandle.GetHiddenLayerCount()[layer + 1];
            }
            else
            {
                dimention1 = TOMLHandle.GetHiddenLayerCount()[layer];
                dimention2 = TOMLHandle.GetOutputLayerCount();
            }

            weights = Matrix<double>.Build.Dense(dimention2, dimention1);

        }
        void SetBiasDimentions(int layer)
        {
            int dimention2;

            if(layer == TOMLHandle.LayerCount)
            {
                dimention2 = TOMLHandle.GetHiddenLayerCount()[layer + 1];
            }
            else
            {
                dimention2 = TOMLHandle.GetOutputLayerCount();
            }

            BiasVector = Vector<double>.Build.Dense(dimention2);

        }

        public List<Matrix<double>> getKernel()
        {
            List<Matrix<double>> Data = new List<Matrix<double>>();

            string directory = $"Data\\CNNLayer";

            foreach(string file in Directory.GetFiles(directory))
            {
                string content = File.ReadAllText(file);
                string[] vals = content.Split(',');

                double[,] doubleArray = new double[(int)Math.Sqrt(vals.Length),(int)Math.Sqrt(vals.Length)];
                int k = 0;
                for(int i = 0; i < Math.Sqrt(vals.Length); i++)
                {
                    for(int j = 0; j < Math.Sqrt(vals.Length); j++)
                    {
                        doubleArray[i, j] = Convert.ToDouble(vals[k]);
                        k++;
                    }
                }

                Data.Add(Matrix<double>.Build.DenseOfArray(doubleArray));
            }
            return Data;
        }
    }
}
