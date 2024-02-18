using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UpdatedProject
{
    internal class CNNLayers
    {

        public Matrix<double> CNNOutput()
        {

            return null;
        }


        public static Matrix<double> Convolution(Matrix<double> image, Matrix<double> kernel)
        {
            int imageWidth = image.ColumnCount;
            int imageHeight = image.RowCount;
            int kernelWidth = kernel.ColumnCount;
            int kernelHeight = kernel.RowCount;

            int resultWidth = imageWidth - kernelWidth + 1;
            int resultHeight = imageHeight - kernelHeight + 1;

            Matrix<double> result = Matrix<double>.Build.Dense(resultHeight, resultWidth);

            for (int i = 0; i < resultWidth; i++)
            {
                for (int j = 0; j < resultHeight; j++)
                {
                    double sum = 0;

                    for (int m = 0; m < kernelWidth; m++)
                    {
                        for (int n = 0; n < kernelHeight; n++)
                        {
                            // Apply convolution operation
                            sum += image[j + n, i + m] * kernel[n, m];
                        }
                    }

                    result[j, i] = sum;
                }
            }

            return result;
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

        public static Matrix<double> SobelXFilter()
        {
            return Matrix<double>.Build.DenseOfArray(new double[,]
            {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
            });
        }

        public static Matrix<double> SobelYFilter()
        {
            return Matrix<double>.Build.DenseOfArray(new double[,]
            {
            { -1, -2, -1 },
            {  0,  0,  0 },
            {  1,  2,  1 }
            });
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
