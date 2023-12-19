using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UpdatedProject
{
    internal class ForwardPass
    {

        public Vector<double> output;

        GetData getdata = new GetData();

        public ForwardPass()
        {
            
        }

        public void Forwards(Matrix<double> weights, Vector<double> LayerVector,Vector<double> Bias,int Pass ,int Layer, int LayerCount)
        {
            string filename = "output";
            
            Vector<double> output = weights * LayerVector + Bias;

            Console.WriteLine(output.ToString());
            Normaliser(output);
            Console.WriteLine(output.ToString());
            
            Layer++; //indexes to the next layer in the network

            weights = getdata.GetWeight(Pass ,Layer);
            Bias = getdata.getBias(Layer);



            getdata.SaveLayorVectors(output, Layer);
            

            if(LayerCount != 0)
            {
                Forwards(weights, Bias, output,Pass ,Layer, LayerCount--);
            }
            else
            {
                File.WriteAllText(filename, string.Join(",", output));
            }

        }

        void Normaliser(Vector<double> output)
        {
            ReLU(output);
        }

        public static void ReLU(Vector<double> x)
        {
            for (int i = 0; i < x.Count(); i++)
            {
                x[i] = Math.Max(0, x[i]);
            }
        }
    }
}
