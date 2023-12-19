using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using MathNet.Numerics.LinearAlgebra;

namespace UpdatedProject
{
    internal class GetData
    {
        private const int Xweight = 784;
        private const int Yweight = 16;
        NetInIt netinit = new NetInIt();


        public Matrix<double> GetWeight(int Pass ,int Layer)
        {
            string Filename = "layer " + Layer + "\\Weights.txt";
            int k = 0;
            Matrix<double> weights = Matrix<double>.Build.DenseOfArray(new double[Yweight, Xweight]);


            if (File.Exists(Filename))
            {
                Console.WriteLine("weights exist");
                string content = File.ReadAllText(Filename);
                string[] vals = content.Split(',');
                for (int i = 0; i < Yweight - 1; i++)
                {
                    for (int j = 0; j < Xweight - 1; j++)
                    {
                        weights[i, j] = Convert.ToDouble(vals[k]);
                        k++;

                    }
                }
                Console.WriteLine(string.Join(",", weights));
                return weights;
            }
            else
            {
                Console.WriteLine("Weights dont exist \n remaking file...");
                
                netinit.WeightGen(Pass, Layer);
                GetWeight(Pass ,Layer);
                return weights;

            }
        }

        public Vector<double> getBias(int Layer)
        {
            string filename = "layer " + Layer + "\\Bias.txt";
            int k = 0;
            Vector<double> BiasVector = Vector<double>.Build.DenseOfArray(new Double[Yweight]);


            if (File.Exists(filename))
            {
                Console.WriteLine("bias exist");

                string content = File.ReadAllText(filename);
                string[] vals = content.Split(',');
                for (int i = 0; i < Yweight; i++)
                {
                    BiasVector[i] = Convert.ToDouble(vals[k]);
                    k++;
                }
                return BiasVector;
            }
            else
            {
                Console.WriteLine("Bias does not exist \n remaking file...");
                netinit.BiasGen(Layer);
                getBias(Layer);
                return BiasVector;
            }
        }

        public void SaveLayorVectors(Vector<double> LayerVector, int layer)
        {
            string filename = $"layer {layer}\\LayerVector.txt";

            File.WriteAllText(filename, string.Join(",", LayerVector));
        }

    }

    
    
}
