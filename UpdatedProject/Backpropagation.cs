using MathNet.Numerics.LinearAlgebra;
using System;
using System.IO;


namespace UpdatedProject
{
    internal class Backpropagation
    {
        ImageHandle label = new ImageHandle();
        public GetData getData = new GetData();


        void BackProp(Vector<double>[] Input, Matrix<double> Weights, Vector<double> Bias, Vector<double> Output, Vector<double> ActualVal, int LearningRate, int layer)
        {
            var error = Output - ActualVal;
            var DeltaOutput = error.PointwiseMultiply(SigmoidDerivative(ActualVal));

            double DeltaWeights = (Input[layer - 1] * (LearningRate) * DeltaOutput);
            Weights -= DeltaWeights;


            Bias = Bias.Subtract(LearningRate * DeltaOutput.Sum());





            var hiddenError = DeltaOutput * Weights.Transpose() * SigmoidDerivative(Input[layer - 1]);
            Weights = getData.GetWeight(layer - 1);
            var DeltaHidden = hiddenError * sigmoid(Input[layer - 1]);
            Weights = Weights - (LearningRate * Input[layer - 1] * DeltaHidden);

            Bias = Bias.Subtract(LearningRate * DeltaHidden.Sum());
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
