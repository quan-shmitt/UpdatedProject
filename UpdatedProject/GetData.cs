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


                string content = File.ReadAllText(filename);
                string[] vals = content.Split(',');
                for (int i = 0; i < BiasVector.Count; i++)
                {
                    BiasVector[i] = Convert.ToDouble(vals[k]);
                    k++;
                }
                return BiasVector;
            }
            else
            {
                Console.WriteLine("Bias does not exist \n remaking file...");
                netinit.BiasGen(layer);
                getBias(layer);
                return BiasVector;
            }
        }

        public Vector<double> LayerVectorGen(int Pass)
        {
            ImageHandle imageHandle = new ImageHandle();

            string filename = $"image_{Pass}_";

            Vector<double> LayerVector = imageHandle.NormRGB(filename, Pass);
            return LayerVector;
        }

        public void SaveWeights(Matrix<double> weights, int layer)
        {
            string filename = $"Data\\Layer {layer}\\Weights.txt";
            File.WriteAllText(filename, string.Join(",", weights));
        }
        
        public void SaveBias(Vector<double> Bias, int layer)
        {
            string filename = $"Data\\Layer {layer}\\Bias.txt";
            File.WriteAllText(filename, string.Join(",", Bias));
        }

        public void SaveLayorVectors(Vector<double> LayerVector,int Pass ,int layer)
        {
            string filename = $"Data\\Pass {Pass}\\Output\\LayerVector.txt";

            File.WriteAllText(filename, string.Join(",", LayerVector));
        }


        public int GetDimentions(int layer)
        {
            string filePath = "Dimentions.txt";
            string[] Data = File.ReadAllLines(filePath);


            string pattern = @"layer\d+Dimention\s*=\s*(\d+)";
            Regex regex = new Regex(pattern);

            Console.Write(layer);
            Match match = regex.Match(Data[layer]);

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

        void SetWeightDimentions(int layer)
        {
            GetData getData = new GetData();

            var dimention1 = getData.GetDimentions(layer);
            var dimention2 = getData.GetDimentions(layer + 1);


            weights = Matrix<double>.Build.Dense(dimention2, dimention1);

        }
        void SetBiasDimentions(int layer)
        {
            GetData getData = new GetData();

            var dimention2 = getData.GetDimentions(layer + 1);

            BiasVector = Vector<double>.Build.Dense(dimention2);

        }



    }
}
