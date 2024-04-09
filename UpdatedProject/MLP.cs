using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;

namespace UpdatedProject
{
    internal class MLP
    {
        public List<Vector<double>> Cache = new List<Vector<double>>();

        readonly ManageData ManageData = new ManageData();

        public MLP()
        {

        }

        public void Forwards(Vector<double> LayerVector, int Layer, int LayerCount)
        {
            Cache.Add(LayerVector);

            Matrix<double> weights = ManageData.GetWeight(Layer);
            Vector<double> Bias = ManageData.getBias(Layer);

            LayerCount--;
            Layer++; //indexes to the next layer in the network

            Vector<double> output = weights * LayerVector + Bias;


            if (LayerCount > 0)
            {
                output = ReLU(output);
            }


            if (LayerCount != 0)
            {
                Forwards(output, Layer, LayerCount);
            }
            else
            {
                output = Softmax(output);
                Cache.Add(output);
            }
        }

        public static Matrix<double> VectorToMatrix(Vector<double> vector, int width, int height)
        {
            Matrix<double> matrix = Matrix<double>.Build.Dense(height, width);

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    matrix[i, j] = vector[i * width + j];
                }
            }

            return matrix;
        }

        public static Vector<double> MatrixToVector(Matrix<double> matrix)
        {
            return Vector<double>.Build.DenseOfArray(matrix.ToColumnMajorArray());
        }

        static public Vector<double> Softmax(Vector<double> logits)
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

        public static Vector<double> ReLU(Vector<double> x)
        {
            for (int i = 0; i < x.Count(); i++)
            {
                x[i] = 1.0 / (1.0 + Math.Exp(-x[i]));
            }
            return x;
        }
    }
}
