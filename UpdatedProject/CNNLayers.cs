using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UpdatedProject
{
    internal class CNNLayers
    {
        //public List<Matrix<double>> Kernel = ;


        public Matrix<double> CNNOutput()
        {
            return null;
        }


        Bitmap ApplySobelFilterWithThreshold(Bitmap inputImage, int threshold)
        {
            // Define Sobel filter kernels for horizontal and vertical directions
            Matrix<int> horizontalKernel = Matrix<int>.Build.DenseOfArray(new int[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } });
            Matrix<int> verticalKernel = Matrix<int>.Build.DenseOfArray(new int[,] { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } });

            // Apply convolution with Sobel kernels
            Bitmap result = ConvolutionFilter(inputImage, horizontalKernel);
            result = ConvolutionFilter(result, verticalKernel);

            // Apply threshold to the magnitude of gradients
            result = ApplyThreshold(result, threshold);

            return result;
        }

        Bitmap ApplyThreshold(Bitmap inputImage, int threshold)
        {
            int width = inputImage.Width;
            int height = inputImage.Height;

            Bitmap result = new Bitmap(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = inputImage.GetPixel(x, y);

                    // Calculate the magnitude of the gradient
                    int magnitude = (int)Math.Sqrt(pixel.R * pixel.R + pixel.G * pixel.G + pixel.B * pixel.B);

                    // Apply threshold
                    if (magnitude < threshold)
                    {
                        result.SetPixel(x, y, Color.Black);
                    }
                    else
                    {
                        result.SetPixel(x, y, Color.White); // You can adjust this color if needed
                    }
                }
            }

            return result;
        }

        Bitmap ConvolutionFilter(Bitmap inputImage, Matrix<int> kernel)
        {
            int width = inputImage.Width;
            int height = inputImage.Height;

            Bitmap result = new Bitmap(width, height);

            int kernelSize = kernel.RowCount; // Assuming square kernel
            int offset = kernelSize / 2;

            Matrix<int> grayImage = Matrix<int>.Build.DenseOfArray(new int[width, height]);

            // Convert input image to grayscale and store it in grayImage matrix
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = inputImage.GetPixel(x, y);
                    int grayValue = (int)(pixel.R * 0.3 + pixel.G * 0.59 + pixel.B * 0.11);
                    grayImage[x, y] = grayValue;
                }
            }

            // Apply convolution operation
            for (int y = offset; y < height - offset; y++)
            {
                for (int x = offset; x < width - offset; x++)
                {
                    int newColorX = 0;
                    int newColorY = 0;

                    for (int ky = -offset; ky <= offset; ky++)
                    {
                        for (int kx = -offset; kx <= offset; kx++)
                        {
                            newColorX += grayImage[x + kx, y + ky] * kernel[kx + offset, ky + offset];
                            newColorY += grayImage[x + kx, y + ky] * kernel[ky + offset, kx + offset];
                        }
                    }

                    int magnitude = (int)Math.Sqrt(newColorX * newColorX + newColorY * newColorY);
                    magnitude = Math.Max(0, Math.Min(255, magnitude));

                    // Set the new pixel value in the result image
                    result.SetPixel(x, y, Color.FromArgb(magnitude, magnitude, magnitude));
                }
            }

            return result;
        }

        Bitmap ResizeImage(Bitmap originalImage, int scaleFactor)
        {
            int newWidth = originalImage.Width * scaleFactor;
            int newHeight = originalImage.Height * scaleFactor;

            Bitmap resizedImage = new Bitmap(newWidth, newHeight);

            for (int x = 0; x < newWidth; x++)
            {
                for (int y = 0; y < newHeight; y++)
                {
                    float originalX = x / (float)scaleFactor;
                    float originalY = y / (float)scaleFactor;

                    Color interpolatedColor = BilinearInterpolation(originalImage, originalX, originalY);
                    resizedImage.SetPixel(x, y, interpolatedColor);
                }
            }

            return resizedImage;
        }

        Color BilinearInterpolation(Bitmap image, float x, float y)
        {
            int x1 = (int)x;
            int y1 = (int)y;
            int x2 = Math.Min(x1 + 1, image.Width - 1);
            int y2 = Math.Min(y1 + 1, image.Height - 1);

            Color q11 = image.GetPixel(x1, y1);
            Color q21 = image.GetPixel(x2, y1);
            Color q12 = image.GetPixel(x1, y2);
            Color q22 = image.GetPixel(x2, y2);

            float dx = x - x1;
            float dy = y - y1;

            int red = (int)(q11.R * (1 - dx) * (1 - dy) + q21.R * dx * (1 - dy) + q12.R * (1 - dx) * dy + q22.R * dx * dy);
            int green = (int)(q11.G * (1 - dx) * (1 - dy) + q21.G * dx * (1 - dy) + q12.G * (1 - dx) * dy + q22.G * dx * dy);
            int blue = (int)(q11.B * (1 - dx) * (1 - dy) + q21.B * dx * (1 - dy) + q12.B * (1 - dx) * dy + q22.B * dx * dy);

            return Color.FromArgb(red, green, blue);
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
