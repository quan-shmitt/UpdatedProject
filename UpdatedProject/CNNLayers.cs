using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace UpdatedProject
{
    internal class CNNLayers
    {
        readonly ManageData manageData = new ManageData();

        public List<Matrix<double>> Kernel;

        public Matrix<double> result;

        Matrix<double> input;

        public string[] algorithms = TOMLHandle.GetCNNStruct();

        int threshold = 350;
        int[] TargetRes = TOMLHandle.GetTargetResolution();
        int poolSize = TOMLHandle.GetPoolSize();



        public CNNLayers(Matrix<double> input)
        {
            Kernel = manageData.getKernel();
            this.input = input;
        }

        public Matrix<double> CNNOutput()
        {
            return null;
        }


        Dictionary<string, MethodInfo> algorithmFunctions = new Dictionary<string, MethodInfo>
        {
            {"ResizeImage", typeof(CNNLayers).GetMethod("ResizeImage") },
            {"ApplyConvolutionFilter", typeof(CNNLayers).GetMethod("ApplyConvolutionFilter") },
            {"ApplyMaxPooling", typeof(CNNLayers).GetMethod("ApplyMaxPooling")},
            {"BilinearInterpolation", typeof(CNNLayers).GetMethod("BilinearInterpolation")}
        };

        static object GetParameterValue(Type parameterType, int layer, Matrix<double> Result)
        {
            // Define your logic to generate parameter values dynamically based on parameterType
            // Here we're just providing some hardcoded values for demonstration purposes
            if (parameterType == typeof(int))
            {
                return layer;
            }
            else if (parameterType == typeof(Matrix<double>))
            {
                return Result;
            }
            else
            {
                // Handle other parameter types as needed
                throw new ArgumentException("Unsupported parameter type");
            }
        }


        public void Forwards(int Layer ,int layerCount, int threashold)
        {

            Matrix<double> LayerMatrix = input;
            result = LayerMatrix;

            CNNLayers cnnLayersInstance = new CNNLayers(input); // Create an instance of CNNLayers

            foreach (string algorithm in algorithms)
            {
                if (algorithmFunctions.ContainsKey(algorithm))
                {
                    ParameterInfo[] parametersInfo = algorithmFunctions[algorithm].GetParameters();

                    // Create an array to hold the parameters
                    object[] parameters = new object[parametersInfo.Length];
                    for (int i = 0; i < parametersInfo.Length; i++)
                    {
                        // Dynamically generate parameter values based on parameter type
                        parameters[i] = GetParameterValue(parametersInfo[i].ParameterType, Layer, result);
                    }

                    // Invoke the method on the instance of CNNLayers
                    algorithmFunctions[algorithm].Invoke(cnnLayersInstance, parameters);
                }
            }


            layerCount--;
            Layer++;

            if (layerCount != 0)
            {
                Forwards(Layer, layerCount, threashold);
            }
            else
            {
                result = ResizeImage(result, TargetRes[0], TargetRes[1]);
            }
        }

        public Matrix<double> ApplyConvolutionFilter(Matrix<double> inputImage, int layer)
        {
            // Apply convolution with Sobel kernels
            Matrix<double> result = ConvolutionFilter(inputImage, Kernel[layer]);
         
            // Apply threshold to the magnitude of gradients
            result = ApplyThreshold(result, threshold);

            return result;
        }

        public Matrix<double> ApplyThreshold(Matrix<double> inputMatrix, int threshold)
        {
            int width = inputMatrix.ColumnCount;
            int height = inputMatrix.RowCount;

            Matrix<double> result = Matrix<double>.Build.Dense(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double magnitude = inputMatrix[y, x];

                    // Apply threshold
                    if (magnitude < threshold)
                    {
                        result[x, y] = 0.0; // Assuming black in a grayscale image
                    }
                    else
                    {
                        result[x, y] = 255.0; // Assuming white in a grayscale image
                    }
                }
            }

            return result;
        }

        public Matrix<double> ConvolutionFilter(Matrix<double> inputMatrix, Matrix<double> kernel)
        {
            int width = inputMatrix.ColumnCount;
            int height = inputMatrix.RowCount;

            Matrix<double> result = Matrix<double>.Build.Dense(width, height);

            int kernelSize = kernel.RowCount; // Assuming square kernel
            int offset = kernelSize / 2;

            // Apply convolution operation
            for (int y = offset; y < height - offset; y++)
            {
                for (int x = offset; x < width - offset; x++)
                {
                    double newColorX = 0;
                    double newColorY = 0;

                    for (int ky = -offset; ky <= offset; ky++)
                    {
                        for (int kx = -offset; kx <= offset; kx++)
                        {
                            newColorX += inputMatrix[y + ky, x + kx] * kernel[kx + offset, ky + offset];
                            newColorY += inputMatrix[y + ky, x + kx] * kernel[ky + offset, kx + offset];
                        }
                    }

                    int magnitude = (int)Math.Sqrt(newColorX * newColorX + newColorY * newColorY);
                    magnitude = Math.Max(0, Math.Min(255, magnitude));

                    // Set the new pixel value in the result matrix
                    result[x, y] = magnitude;
                }
            }

            return result;
        }

        public Matrix<double> ApplyMaxPooling(Matrix<double> inputMatrix)
        {
            int width = inputMatrix.ColumnCount;
            int height = inputMatrix.RowCount;

            // Calculate the new dimensions after max pooling
            int newWidth = width / poolSize;
            int newHeight = height / poolSize;

            Matrix<double> result = Matrix<double>.Build.Dense(newHeight, newWidth);

            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    int startX = x * poolSize;
                    int startY = y * poolSize;

                    // Find the maximum value in the pooling window
                    double maxVal = GetMaxValueInWindow(inputMatrix, startX, startY, poolSize);

                    // Set the maximum value in the result matrix
                    result[y, x] = maxVal;
                }
            }

            return result;
        }

        public double GetMaxValueInWindow(Matrix<double> matrix, int startX, int startY, int poolSize)
        {
            double maxVal = double.MinValue;

            for (int i = 0; i < poolSize; i++)
            {
                for (int j = 0; j < poolSize; j++)
                {
                    double currentValue = matrix[startY + i, startX + j];
                    maxVal = Math.Max(maxVal, currentValue);
                }
            }

            return maxVal;
        }

        public Matrix<double> ResizeImage(Matrix<double> originalMatrix, int targetWidth, int targetHeight)
        {
            int originalWidth = originalMatrix.ColumnCount;
            int originalHeight = originalMatrix.RowCount;

            // Calculate scaling factors for width and height
            double widthScaleFactor = (double)targetWidth / originalWidth;
            double heightScaleFactor = (double)targetHeight / originalHeight;

            Matrix<double> resizedMatrix = Matrix<double>.Build.Dense(targetHeight, targetWidth);

            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    // Calculate original coordinates based on scaling factors
                    float originalX = (float)x / (float)widthScaleFactor;
                    float originalY = (float)y / (float)heightScaleFactor;

                    double interpolatedValue = BilinearInterpolation(originalMatrix, originalX, originalY);
                    resizedMatrix[y, x] = interpolatedValue;
                }
            }

            return resizedMatrix;
        }

        public double BilinearInterpolation(Matrix<double> matrix, float x, float y)
        {
            int xFloor = (int)Math.Floor(x);
            int yFloor = (int)Math.Floor(y);
            int xCeiling = (int)Math.Ceiling(x);
            int yCeiling = (int)Math.Ceiling(y);

            double q11 = matrix[yFloor, xFloor];
            double q12 = matrix[yCeiling, xFloor];
            double q21 = matrix[yFloor, xCeiling];
            double q22 = matrix[yCeiling, xCeiling];

            double xLerp = x - xFloor;
            double yLerp = y - yFloor;

            double topInterpolation = q11 * (1 - xLerp) + q21 * xLerp;
            double bottomInterpolation = q12 * (1 - xLerp) + q22 * xLerp;

            double finalInterpolation = topInterpolation * (1 - yLerp) + bottomInterpolation * yLerp;

            return finalInterpolation;
        }


        public Matrix<double> VectorToMatrix(Vector<double> vector, int width, int height)
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

        public Vector<double> MatrixToVector(Matrix<double> matrix)
        {
            return Vector<double>.Build.DenseOfArray(matrix.ToColumnMajorArray());
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
