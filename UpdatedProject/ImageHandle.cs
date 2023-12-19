using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace UpdatedProject
{
    internal class ImageHandle
    {
        private const int Xweight = 784;

        public Vector<double> NormRGB(string path, int index)
        {

            Console.WriteLine(path);
            Vector<double> RGBVal = Vector<double>.Build.DenseOfArray(new Double[Xweight]);

            try
            {
                string[] fileNames = Directory.EnumerateFiles("images")
                                       .OrderBy(filename => ExtractNumberFromFilename(filename))
                                       .ToArray();


                string filenameToCheck = $"image_{index}_";

                Console.WriteLine("checking for file...");
                if (fileNames.Any(fileName => fileName.Contains(filenameToCheck)))
                {
                    Console.WriteLine($"file exists {filenameToCheck}");
                    Console.WriteLine($"file exists {fileNames[index]}");
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
