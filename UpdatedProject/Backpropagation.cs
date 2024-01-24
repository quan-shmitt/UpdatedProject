using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;


namespace UpdatedProject
{
    internal class Backpropagation
    {
        ImageHandle label = new ImageHandle();
        public GetData getData = new GetData();

        List<Matrix<double>> Weights = new List<Matrix<double>>();
        List<Vector<double>> Bias = new List<Vector<double>>(); 

        public Backpropagation(int layer)
        {

            for (int i = 0; i < layer; i++)
            {
                Weights.Add(getData.GetWeight(i));
            }

            for (int i = 0; i < layer; i++)
            {
                Bias.Add(getData.getBias(i));
            }
        }

        public void BackProp(List<Vector<double>> Input, Vector<double> Output, double LearningRate, int layer)
        {

            var error = Output - Input[layer];
            var DeltaOutput = error.PointwiseMultiply(SigmoidDerivative(Input[layer]));

            double DeltaWeights = (Input[layer - 1] * (LearningRate) * DeltaOutput);
            Weights[layer] -= DeltaWeights;


            Bias[layer] -=(LearningRate * DeltaOutput.Sum());
            getData.SaveWeights(Weights[layer], layer);
            getData.SaveBias(Bias[layer], layer);

            layer--;

            while (layer > 0)
            {
                var hiddenError = DeltaOutput * Weights[layer].Transpose() * SigmoidDerivative(Input[layer]);
                var DeltaHidden = hiddenError * sigmoid(Input[layer]);
                Weights[layer] -= (LearningRate * Input[layer] * DeltaHidden);

                Bias[layer] -= (LearningRate * DeltaHidden.Sum());

                getData.SaveWeights(Weights[layer], layer);
                getData.SaveBias(Bias[layer], layer);

                layer--;
            }
        }


        Vector<double> sigmoid(Vector<double> x)
        {
            Vector<double> y = Vector<double>.Build.DenseOfArray(new double[x.Count]);
            foreach (var v in x)
            {
                y.Add(1 / (1 + Math.Exp(-v)));
            }
            return y;
        }

        Vector<double> SigmoidDerivative(Vector<double> x)
        {
            Vector<double> y = Vector<double>.Build.DenseOfArray(new double[x.Count]);
            var negative = sigmoid(x);

            foreach (var v in negative)
            {
                y.Add(1 - v);
            }

            return negative.PointwiseMultiply(y);

        }
    }
}
