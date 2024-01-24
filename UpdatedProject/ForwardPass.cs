using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace UpdatedProject
{
    internal class ForwardPass
    {
        public List<Vector<double>> Cache = new List<Vector<double>>();

        Vector<double> output;
        public Vector<double> LayerVector;

        GetData getdata = new GetData();

        public ForwardPass(int Pass, int Layers)
        {
            Cache = new List<Vector<double>>();
            LayerVector = getdata.LayerVectorGen(Pass);

        }

        public void Forwards(Vector<double> LayerVector ,int Pass ,int Layer, int LayerCount)
        {
            Cache.Add(LayerVector);
            

            Matrix<double> weights = getdata.GetWeight(Layer);
            Vector<double> Bias = getdata.getBias(Layer);

            LayerCount--;
            Layer++; //indexes to the next layer in the network
            
            output = weights * LayerVector + Bias;

            Normaliser(output);
            


            if(LayerCount != 0 ) 
            {
                weights = getdata.GetWeight(Layer);
                Bias = getdata.getBias(Layer);
                Forwards(output, Pass, Layer, LayerCount);

            }
            getdata.SaveLayorVectors(output, Pass, Layer);

        }

        void Normaliser(Vector<double> output)
        {
            ReLU(output);
        }

        public static void ReLU(Vector<double> x)
        {
            for (int i = 0; i < x.Count(); i++)
            {
                x[i] = 1.0 / (1.0 + Math.Exp(-x[i]));
            }
        }
    }
}
