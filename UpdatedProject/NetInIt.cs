using MathNet.Numerics.Integration;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.IO;
using System.Reflection.Emit;
using System.Runtime.InteropServices;

namespace UpdatedProject
{
    internal class NetInIt
    {

        public static ImageHandle imageHandle = new ImageHandle();

        public Matrix<double> weights;

        public Matrix<double> LayerMatrix;


        public Vector<double> BiasVector;


        public NetInIt(int Pass, int layerCount, int CNNCount)
        {

            //CreateParentDataset(Convert.ToUInt32(MaxImageCount));
            int[] dim = new int[] {TOMLHandle.GetKernelSize(), TOMLHandle.GetKernelSize()};

            LayerMatrix = new ManageData().LayerVectorGen(Pass);

            fileGen(layerCount, CNNCount, dim);

            LayerGen(Pass, layerCount - 1);

        }



        public NetInIt()
        {

        }
        //heheheha


        static public int GetFileDimentions(int Pass)
        {
            string filename = $"image_{Pass}_";
            string DimFile = "Dimentions.txt";


            Matrix<double> vals = imageHandle.NormRGB(filename, Pass);



            string existingText;
            using (StreamReader reader = new StreamReader(DimFile))
            {
                existingText = reader.ReadToEnd();
            }

            // Write the new text and then the existing content back to the file
            using (StreamWriter writer = new StreamWriter(DimFile))
            {
                writer.Write($"layer0Dimention = {vals.ColumnCount * vals.RowCount}");
                writer.Write(existingText);
            }

            return vals.ColumnCount * vals.RowCount;
        }

        void fileGen(int layer, int CNNlayer, int[] dim)
        {
            MLPGen(layer);
            CNNGen(CNNlayer, dim);
        }

        public void MLPGen(int layer)
        {
            for (int i = 0; i < layer; i++)
            {
                string filename = $"Data\\Layer {i}";
                if (!File.Exists(filename))
                {
                    Directory.CreateDirectory(filename);
                }
            }
        }
        void LayerGen(int Pass, int layer)
        {
            for (int i = 0; i <= layer; i++)
            {
                WeightGen(i);
                BiasGen(i);
            }
        }


        public void CNNGen(int CNNLayer, int[] dim)
        {
            GaussianRandomGenerator gaussianRandomGenerator = new GaussianRandomGenerator();

            string DirectoryName = $"Data\\CNNLayer";

            if(!Directory.Exists(DirectoryName))
            {
                Directory.CreateDirectory(DirectoryName);
            }

            for (int i = 0; i < CNNLayer; i++)
            {
                string filename = $"Data\\CNNLayer\\Kernel {i}.txt";
                if (!File.Exists(filename))
                {
                    File.WriteAllText(filename, string.Join(",", CNNLayerGen(dim, gaussianRandomGenerator)));
                }
            }
        }

         double[] CNNLayerGen(int[] dimentions, GaussianRandomGenerator gaussianRandomGenerator)
         {
            double[] vals = new double[dimentions[0] * dimentions[1]];

            for (int i = 0;i < dimentions[0] * dimentions[1]; i++)
            {
                vals[i] = gaussianRandomGenerator.Generate(0, 1);
            }

            return vals;
            
         }




        public void WeightGen(int layer)
        {
            SetWeightDimentions(layer);

            GaussianRandomGenerator generator = new GaussianRandomGenerator();

            double mean = 0.0;
            double stdDev = 1.0;
            string Filename = $"Data\\Layer {layer}\\Weights.txt";
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

            }
            else
            {
                string[] vals = new string[weights.ColumnCount * weights.RowCount];
                for (int i = 0; i < weights.ColumnCount * weights.RowCount; i++)
                {
                    vals[i] = Convert.ToString(generator.Generate(mean, stdDev));
                }

                try
                {
                    File.WriteAllText(Filename, string.Join(",", vals));
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"An error occurred: {ex.Message}");

                }




                System.Array.Clear(vals, 0, vals.Length);

                string content = File.ReadAllText(Filename);
                vals = content.Split(',');
                for (int i = 0; i < weights.ColumnCount; i++)
                {
                    for (int j = 0; j < weights.RowCount; j++)
                    {
                        weights[j, i] = Convert.ToDouble(vals[k]);
                        k++;
                    }
                }


            }
        }


        public void BiasGen(int layer)
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
            }
            else
            {
                string[] vals = new string[BiasVector.Count];
                for (int i = 0; i < BiasVector.Count; i++)
                {
                    vals[i] = "0";
                }

                try
                {
                    File.WriteAllText(filename, string.Join(",", vals));
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"An error occurred: {ex.Message}");

                }


                System.Array.Clear(vals, 0, vals.Length);

                string content = File.ReadAllText(filename);
                vals = content.Split(',');
                for (int i = 0; i < BiasVector.Count; i++)
                {
                    BiasVector[i] = Convert.ToDouble(vals[k]);
                    k++;
                }


            }
        }


        void SetWeightDimentions(int layer)
        {
            ManageData getData = new ManageData();

            Console.WriteLine(layer);

            int dimention1;
            int dimention2;

            if (layer == 0) 
            { 
                dimention1 = LayerMatrix.ColumnCount * LayerMatrix.RowCount;

                dimention2 = TOMLHandle.GetHiddenLayerCount()[layer];
            }
            else if(layer < TOMLHandle.LayerCount)
            {
                dimention1 = TOMLHandle.GetHiddenLayerCount()[layer];
                dimention2 = TOMLHandle.GetHiddenLayerCount()[layer + 1];
            }
            else
            {
                dimention1 = TOMLHandle.GetHiddenLayerCount()[layer - 1];
                dimention2 = TOMLHandle.GetOutputLayerCount();
            }

            weights = Matrix<double>.Build.Dense(dimention2, dimention1);

        }
        void SetBiasDimentions(int layer)
        {
            ManageData getData = new ManageData();

            int dimention2;

            if (layer < TOMLHandle.LayerCount)
            {
                dimention2 = TOMLHandle.GetHiddenLayerCount()[layer];
            }
            else
            {
                dimention2 = TOMLHandle.GetOutputLayerCount();
            }
            BiasVector = Vector<double>.Build.DenseOfArray(new double[dimention2]);

        }


    }
    public class GaussianRandomGenerator
    {
        private Random random;

        public GaussianRandomGenerator()
        {
            random = new Random();
        }

        public GaussianRandomGenerator(int seed)
        {
            random = new Random(seed);
        }

        public double Generate(double mean, double stdDev)
        {
            // Generate two random numbers between 0 and 1
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();

            // Apply the Box-Muller transform to generate a standard Gaussian random number
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

            // Scale and shift the standard Gaussian random number to have the desired mean and stdDev
            double result = mean + z * stdDev;

            return result;
        }

    }

}
