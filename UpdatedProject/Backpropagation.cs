using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace UpdatedProject
{
    internal class Backpropagation
    {
        readonly ImageHandle label = new ImageHandle();
        public ManageData getData = new ManageData();

        readonly List<Matrix<double>> Weights = new List<Matrix<double>>();
        readonly List<Vector<double>> Bias = new List<Vector<double>>();

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

        public void BackProp(List<Vector<double>> LayerVectors, Vector<double> Target, double LearningRate, int layer)
        {
            Program.cost += CalculateSparseCategoricalCrossEntropy(LayerVectors[layer], Convert.ToInt32(Target.MaximumIndex()));

            Matrix<double> gradientWrtWeights = (LayerVectors[layer] - Target).ToColumnMatrix() * LayerVectors[layer - 1].ToRowMatrix();

            Vector<double> gradientWrtBias = LayerVectors[layer] - Target;



            Weights[layer - 1] -= LearningRate * gradientWrtWeights;


            Bias[layer - 1] -= LearningRate * gradientWrtBias;

            getData.SaveWeights(Weights[layer - 1], layer - 1);
            getData.SaveBias(Bias[layer - 1], layer - 1);

            var UpstreamGradient = LayerVectors[layer] - Target;

            layer--;

            while (layer > 0)
            {
                gradientWrtWeights = ReLU_Derivative(LayerVectors[layer]).ToColumnMatrix() * LayerVectors[layer - 1].ToRowMatrix();

                gradientWrtBias = Vector<double>.Build.DenseOfArray(new double[Bias[layer - 1].Count]);

                var GradWrtLlogits = ReLU_Derivative(LayerVectors[layer]);

                int k = 0;
                for (int i = 0; i < Bias[layer - 1].Count; i++)
                {
                    gradientWrtBias[i] = GradWrtLlogits[i] * UpstreamGradient[k];
                    k++;
                    if (k >= UpstreamGradient.Count)
                    {
                        k = 0;
                    }
                }

                Weights[layer - 1] -= LearningRate * gradientWrtWeights;

                Bias[layer - 1] -= LearningRate * gradientWrtBias;

                getData.SaveWeights(Weights[layer - 1], layer - 1);
                getData.SaveBias(Bias[layer - 1], layer - 1);

                layer--;
            }
        }

        public void CNNBackProp(Matrix<double> kernel)
        {

        }



        double CalculateSparseCategoricalCrossEntropy(Vector<double> predictedProbabilities, int trueLabel)
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
        static Vector<double> ReLU_Derivative(Vector<double> logits)
        {
            return logits.Map(x => x > 0 ? 1.0 : 0.0);
        }

        Vector<double> Softmax(Vector<double> logits)
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

        Vector<double> SoftmaxDerivativeVector(Vector<double> logits)
        {
            int K = logits.Count;
            Vector<double> result = Vector<double>.Build.Dense(K, (i) =>
            {
                int row = i / K;
                int col = i % K;
                if (row == col)
                    return logits[row] * (1 - logits[row]);
                else
                    return -logits[row] * logits[col];
            });

            return result;
        }




    }
}
