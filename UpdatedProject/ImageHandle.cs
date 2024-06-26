﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace UpdatedProject
{
    internal class ImageHandle
    {
        private const int Xweight = 784;


        public Vector<double> Label(int index, int dimentionSize)
        {
            Vector<double> label = Vector<double>.Build.DenseOfArray(new double[dimentionSize]);


            string pattern = $"image_{index}_label_(\\d+)";

            string[] fileNames = Directory.EnumerateFiles("images")
                                       .OrderBy(filename => ExtractNumberFromFilename(filename))
                                       .ToArray();

            Match match = Regex.Match(fileNames[index], pattern);

            if (match.Success)
            {
                string numberAfterLabel = match.Groups[1].Value;
                label[Convert.ToInt32(numberAfterLabel)] = 1;
                return label;
            }
            else { return label; }
        }



        public Matrix<double> NormRGB(string path, int index)
        {
            string[] fileNames = Directory.EnumerateFiles("images")
                                   .OrderBy(filename => ExtractNumberFromFilename(filename))
                                   .ToArray();

            string filenameToCheck = $"image_{index}_";

            if (fileNames.Any(fileName => fileName.Contains(filenameToCheck)))
            {
                using (Bitmap image = new Bitmap(fileNames[index]))
                {
                    Matrix<double> RGBVal = Matrix<double>.Build.DenseOfArray(new Double[image.Width, image.Height]);

                    for (int y = 0; y < image.Height; y++)
                    {
                        for (int x = 0; x < image.Width; x++)
                        {
                            Color color = image.GetPixel(x, y);

                            double NormColor = color.GetBrightness();

                            RGBVal[x, y] = NormColor;


                        }
                    }
                    return RGBVal;
                }
            }
            else
            {
                return null;
            }
        }

        private int ExtractNumberFromFilename(string filename)
        {
            string numberPart = new string(filename.Where(char.IsDigit).ToArray());
            return int.Parse(numberPart);
        }
    }
}
