using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using static System.Net.Mime.MediaTypeNames;
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



        public Vector<double> NormRGB(string path, int index)
        {



            Vector<double> RGBVal = Vector<double>.Build.DenseOfArray(new Double[Xweight]);

            try
            {
                string[] fileNames = Directory.EnumerateFiles("images")
                                       .OrderBy(filename => ExtractNumberFromFilename(filename))
                                       .ToArray();


                string filenameToCheck = $"image_{index}_";


                if (fileNames.Any(fileName => fileName.Contains(filenameToCheck)))
                {
                    using (Bitmap image = new Bitmap(fileNames[index]))
                    {
                        for (int y = 0; y < image.Height; y++)
                        {
                            for (int x = 0; x < image.Width; x++)
                            {
                                Color color = image.GetPixel(x, y);

                                double NormColor = color.GetBrightness();

                                RGBVal[x + y * image.Width] = NormColor;


                            }
                        }
                    }
                    return RGBVal;
                }
                else
                {
                    Console.WriteLine("file doesnt exist");
                    return RGBVal;
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error: {e.Message}");
                return RGBVal;
            }
        }

        private int ExtractNumberFromFilename(string filename)
        {
            string numberPart = new string(filename.Where(char.IsDigit).ToArray());
            return int.Parse(numberPart);
        }
    }
}
