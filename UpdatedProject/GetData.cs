using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;

namespace UpdatedProject
{

    internal class GetData
    {
        private const int Xweight = 784;
        private const int Yweight = 16;
        NetInIt netinit = new NetInIt();


        public Matrix<double> GetWeight(int Pass, int layer)
        {
            string Filename = $"Pass {Pass}\\layer {layer}\\Weights.txt";
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
                Console.WriteLine("Weights dont exist \n Remaking file...");
                
                netinit.WeightGen(Pass, layer);
                GetWeight(Pass, layer);
                return weights;

            }
        }

        public Vector<double> getBias(int Pass, int layer)
        {
            string filename = $"Pass {Pass}\\layer {layer}\\Bias.txt";
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
                netinit.BiasGen(Pass, layer);
                getBias(Pass, layer);
                return BiasVector;
            }
        }

        public void SaveLayorVectors(Vector<double> LayerVector,int Pass ,int layer)
        {
            string filename = $"Pass {Pass}\\layer {layer}\\LayerVector.txt";

            File.WriteAllText(filename, string.Join(",", LayerVector));
        }


        public int GetDimentions(int layer)
        {
            string filePath = "Dimentions.txt";
            string[] Data = File.ReadAllLines(filePath);

            foreach (string data in Data)
            {

                string pattern = @"layer\d+Dimention\s*=\s*(\d+)";
                Regex regex = new Regex(pattern);

                Match match = regex.Match(data);

                if (match.Success)
                {
                    string dimensionValue = match.Groups[1].Value;
                    if (int.TryParse(dimensionValue, out int extractedDimension))
                    {
                        return extractedDimension;
                    }
                    else
                    {
                        Console.WriteLine("Failed to convert the extracted string to an integer.");
                        return 0;
                    }
                }
                else
                {
                    Console.WriteLine("No match found in the string");
                    return 0;
                }
            }
            return 0;
        }
    }
}
