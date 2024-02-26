using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
using System;
using System.Collections.Generic;
using System.Configuration.Assemblies;
using System.IO;
using System.Linq;
using System.Threading;

namespace UpdatedProject
{
    internal class Backpropagation
    {
        ImageHandle label = new ImageHandle();
        public ManageData getData = new ManageData();

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

        public void BackProp(List<Vector<double>> LayerVectors ,Vector<double> Target, double LearningRate, int layer)
        {
            LayerVectors[layer] = Softmax(LayerVectors[layer]);

            Program.cost += CalculateSparseCategoricalCrossEntropy(LayerVectors[layer], Convert.ToInt32(Target.MaximumIndex()));


            Vector<double> gradientWrtWeights = LayerVectors[layer].PointwiseMultiply(LayerVectors[layer] - Target) * SoftmaxDerivativeMatrix(LayerVectors[layer]);

            Vector<double> gradientWrtLogits = SoftmaxDerivativeMatrix(LayerVectors[layer]) * (LayerVectors[layer] - Target);

            Vector<double> gradientWrtBiases = gradientWrtLogits;

            


            for(int i = 0; i < Weights[layer - 1].RowCount; i++)
            {
                for(int j = 0; j < Weights[layer - 1].ColumnCount; j++)
                {
                    (Weights[layer - 1])[i, j] -= LearningRate * gradientWrtWeights[i];
                }
            }

            Bias[layer - 1] -= LearningRate * gradientWrtBiases;

            getData.SaveWeights(Weights[layer - 1], layer - 1);
            getData.SaveBias(Bias[layer - 1], layer - 1);


            layer--;

            while (layer > 0)
            {
                gradientWrtWeights = LayerVectors[layer].PointwiseMultiply(LayerVectors[layer] - Target) * SoftmaxDerivativeMatrix(LayerVectors[layer]);

                gradientWrtBiases = SoftmaxDerivativeMatrix(LayerVectors[layer]) * (LayerVectors[layer] - Target);

                for (int i = 0; i < Weights[layer - 1].RowCount; i++)
                {
                    for (int j = 0; j < Weights[layer - 1].ColumnCount; j++)
                    {
                        (Weights[layer - 1])[i, j] -= LearningRate * gradientWrtWeights[i];
                    }
                }


                Bias[layer - 1] -= LearningRate * gradientWrtBiases;

                getData.SaveWeights(Weights[layer - 1], layer - 1);
                getData.SaveBias(Bias[layer - 1], layer - 1);

                layer--;
            }
        }

        static double CalculateSparseCategoricalCrossEntropy(Vector<double> predictedProbabilities, int trueLabel)
        {
            if (predictedProbabilities == null || predictedProbabilities.Count == 0)
            {
                throw new ArgumentException("Invalid predicted probabilities array");
            }

            if (trueLabel < 0 || trueLabel >= predictedProbabilities.Count)
            {
                throw new ArgumentException("Invalid true label index");
            }

            double epsilon = 1e-15; // Small value to prevent log(0) errors
            double[] logProbabilities = new double[predictedProbabilities.Count];

            // Calculate log probabilities
            for (int i = 0; i < predictedProbabilities.Count; i++)
            {
                // Use a small epsilon to avoid log(0) errors
                logProbabilities[i] = Math.Log(Math.Max(predictedProbabilities[i], epsilon));
            }

            // Calculate sparse categorical cross entropy loss
            double loss = -logProbabilities[trueLabel];

            return loss;
        }


        Vector<double> ReLU(Vector<double> x)
        {
            return x.PointwiseMaximum(0);
        }

        // Derivative of ReLU activation function
        Vector<double> ReLU_Derivative(Vector<double> x)
        {
            return x.PointwiseSign();
        }

        static Vector<double> Softmax(Vector<double> logits)
        {
            // Avoid numerical instability by subtracting the maximum logit
            double maxLogit = logits.Maximum();
            Vector<double> expLogits = logits.Subtract(maxLogit).PointwiseExp();

            // Calculate the sum of exponentials
            double sumExp = expLogits.Sum();

            // Compute the softmax probabilities
            Vector<double> probabilities = expLogits.Divide(sumExp);

            return probabilities;
        }
        static Matrix<double> SoftmaxDerivativeMatrix(Vector<double> softmax)
        {
            int K = softmax.Count;
            Matrix<double> result = Matrix<double>.Build.Dense(K, K, (i, j) =>
            {
                if (i == j)
                    return softmax[i] * (1 - softmax[i]);
                else
                    return -softmax[i] * softmax[j];
            });

            return result;
        }

    }
}
