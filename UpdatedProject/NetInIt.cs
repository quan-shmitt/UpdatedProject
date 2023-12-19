using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra ;

namespace UpdatedProject
{
    internal class NetInIt
    {
        public static ImageHandle imageHandle = new ImageHandle();

        private const int Xweight = 784;
        private const int Yweight = 16;

        private int layer;


        public Matrix<double> weights = Matrix<double>.Build.DenseOfArray(new double[Yweight, Xweight]);

        public Vector<double> LayerVector = Vector<double>.Build.DenseOfArray(new Double[Xweight]);

        public Vector<double> BiasVector = Vector<double>.Build.DenseOfArray(new Double[Yweight]);

        

        public NetInIt(int Pass, int layer)
        {

            this.layer = layer;

            FileGen(layer, Pass);

            LayerGen(layer, Pass);

        }

        


        public NetInIt()
        {

        }
        //heheheha


        public void FileGen(int Layer, int Pass)
        {
            string filename = "Pass ";

            for (int i = 0; i <= Pass; i++)
            {
                try
                {
                    Directory.CreateDirectory(filename + i);
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Error: {e.Message}");
                }
            }

            filename = "Layer ";

            for (int i = 0; i <= Layer; i++)
            {
                try
                {
                    Directory.CreateDirectory(filename + i);
                    Console.WriteLine("file made");
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Error: {e.Message}");
                }
            }



        }
        void LayerGen(int layer ,int Pass)
        {
            if (Pass == 0)
            {
                WeightGen(layer ,Pass);
                LayerVectorGen(Pass);
                BiasGen(Pass);
            }
            else
            {
                WeightGen(layer ,Pass);
                BiasGen(Pass);
            }
        }



        public void WeightGen(int Pass, int layer)
        {
            GaussianRandomGenerator generator = new GaussianRandomGenerator();


            double mean = 0.0;
            double stdDev = 1.0;
            string Filename = $"Pass {Pass}\\layer {layer} \\Weights.txt";
            int k = 0;



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
            }
            else
            {
                string[] vals = new string[Yweight * Xweight];
                for (int i = 0; i < Yweight * Xweight; i++)
                {
                    vals[i] = Convert.ToString(generator.Generate(mean, stdDev));
                }

                try
                {
                    File.WriteAllText(Filename, string.Join(",", vals));
                    Console.WriteLine("Binary file saved successfully.");
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"An error occurred: {ex.Message}");

                }


                Console.WriteLine("file saved");

                Array.Clear(vals, 0, vals.Length);

                string content = File.ReadAllText(Filename);
                vals = content.Split(',');
                for (int i = 0; i < Yweight - 1; i++)
                {
                    for (int j = 0; j < Xweight - 1; j++)
                    {
                        weights[j, i] = Convert.ToDouble(vals[k]);
                        k++;
                    }
                }
                Console.WriteLine(string.Join(",", weights));

            }
        }


        public void BiasGen(int Pass)
        {
            string filename = "layer " + Pass + "\\Bias.txt";
            int k = 0;

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
            }
            else
            {
                string[] vals = new string[Yweight];
                for (int i = 0; i < Yweight; i++)
                {
                    vals[i] = "0";
                }

                try
                {
                    File.WriteAllText(filename, string.Join(",", vals));
                    Console.WriteLine("File saved successfully");
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"An error occurred: {ex.Message}");

                }
                Console.WriteLine("file saved");

                Array.Clear(vals, 0, vals.Length);

                string content = File.ReadAllText(filename);
                vals = content.Split(',');
                for (int i = 0; i < Yweight; i++)
                {
                    LayerVector[i] = Convert.ToDouble(vals[k]);
                    k++;
                }


            }
        }


        void LayerVectorGen(int layer)
        {
            GaussianRandomGenerator generator = new GaussianRandomGenerator();

            string Filename = "layer " + layer + "\\LayerVectors.txt";
            int k = 0;



            if (File.Exists(Filename))
            {
                Console.WriteLine("node values exist");
                string content = File.ReadAllText(Filename);
                string[] vals = content.Split(',');
                for (int i = 0; i < Xweight - 1; i++)
                {
                    LayerVector[i] = Convert.ToDouble(vals[k]);
                    k++;
                }
                Console.WriteLine(LayerVector.ToString());

            }
            else
            {
                Vector<double> vals = Vector<double>.Build.DenseOfArray(new Double[Xweight]);

                string filename = $"image_{layer}_";

                vals = imageHandle.NormRGB(filename, layer);

                try
                {
                    File.WriteAllText(Filename, string.Join(",", vals));
                    Console.WriteLine("File saved successfully.");
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"An error occurred: {ex.Message}");
                }


                Console.WriteLine("file saved");



                string content = File.ReadAllText(Filename);
                string[] values = content.Split(',');
                for (int i = 0; i < Xweight; i++)
                {
                    LayerVector[i] = Convert.ToDouble(values[k]);
                    k++;
                }
                Console.WriteLine(LayerVector.ToString());


            }
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
