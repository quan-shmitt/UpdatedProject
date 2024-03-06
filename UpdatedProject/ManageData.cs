using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using MathNet.Numerics.LinearAlgebra;
using System.Text.RegularExpressions;
using System.Runtime.InteropServices;
using HDF.PInvoke;
using System.Threading;
using System.Drawing.Text;
using System.Diagnostics.Eventing.Reader;
using System.Runtime.InteropServices.ComTypes;
using System.Drawing;

namespace UpdatedProject
{

    internal class ManageData
    {
        NetInIt netinit = new NetInIt();

        public Matrix<double> weights;
        public Vector<double> BiasVector;


        public Matrix<double> GetWeight(int layer)
        {
            SetWeightDimentions(layer);

            string Filename = $"Data\\layer {layer}\\Weights.txt";
            int k = 0;


            if (File.Exists(Filename))
            {
                lock (Program.filelock)
                {
                    try
                    {
                        while (!Program.isfree)
                        {
                            Monitor.Wait(Program.filelock);
                        }
                        Program.isfree = false;
                        string content = File.ReadAllText(Filename);
                        string[] vals = content.Split(',');
                        for (int i = 0; i < weights.ColumnCount - 1; i++)
                        {
                            for (int j = 0; j < weights.RowCount - 1; j++)
                            {
                                weights[j, i] = Convert.ToDouble(vals[k]);
                                k++;

                            }
                        }
                        return weights;

                    }
                    finally
                    {
                        Program.isfree = true;
                        Monitor.Pulse(Program.filelock);
                    }
                }
            }
            else
            {
                Console.WriteLine("Weights dont exist \n Remaking file...");
                
                netinit.WeightGen(layer);
                GetWeight(layer);
                return weights;

            }
        }

        public Vector<double> getBias(int layer)
        {
            SetBiasDimentions(layer);

            string filename = $"Data\\layer {layer}\\Bias.txt";
            int k = 0;

            if (File.Exists(filename))
            {
                lock (Program.filelock)
                {
                    try
                    {
                        while (!Program.isfree)
                        {
                            Monitor.Wait(Program.filelock);
                        }
                        Program.isfree = false;

                        string content = File.ReadAllText(filename);
                        string[] vals = content.Split(',');
                        for (int i = 0; i < BiasVector.Count; i++)
                        {
                            BiasVector[i] = Convert.ToDouble(vals[k]);
                            k++;
                        }
                        return BiasVector;
                    }
                    finally
                    {
                        Program.isfree = true;
                        Monitor.Pulse(Program.filelock);
                    }
                }
            }
            else
            {
                Console.WriteLine("Bias does not exist \n remaking file...");
                netinit.BiasGen(layer);
                getBias(layer);
                return BiasVector;
            }
        }

        public Vector<double> GetImage()
        {
            string FileName = "Data\\ImageToProcess";

            if (Directory.Exists(FileName))
            {
                Console.WriteLine("file exists");
                string[] files = Directory.GetFiles(FileName);
                string file = files[0];
                using (Bitmap Image = new Bitmap(file))
                {
                    Vector<double> RGBVal = Vector<double>.Build.DenseOfArray(new double[Image.Width * Image.Height]);

                    for (int y = 0; y < Image.Height; y++)
                    {
                        for (int x = 0; x < Image.Width; x++)
                        {
                            Color color = Image.GetPixel(x, y);

                            double NormColor = color.GetBrightness();

                            RGBVal[x + y * Image.Width] = NormColor;

                        }
                    }
                    return RGBVal;
                }
            }
            else
            {
                Console.WriteLine("File doesnt exist, remaking File...");

                Directory.CreateDirectory(FileName);
                return GetImage();
            }
            
        }


        public Matrix<double> LayerVectorGen(int Pass)
        {
            ImageHandle imageHandle = new ImageHandle();

            string filename = $"image_{Pass}_";

            Matrix<double> LayerVector = imageHandle.NormRGB(filename, Pass);
            return LayerVector;
        }

        public void SaveWeights(Matrix<double> weights, int layer)
        {


            string filename = $"Data\\Layer {layer}\\Weights.txt";

            var weight = new string[weights.ColumnCount * weights.RowCount];

            int k = 0;

            for(int i = 0; i < weights.ColumnCount; i++)
            {
                for(int j = 0; j < weights.RowCount; j++)
                {
                    weight[k] = weights[j, i].ToString();
                    k++;
                }
            }

            lock (Program.filelock)
            {
                try
                {
                    while (!Program.isfree)
                    {
                        Monitor.Wait(Program.filelock);
                    }
                    Program.isfree = false;

                    File.WriteAllText(filename, string.Join(",", weight));

                }
                finally
                {
                    Program.isfree = true;
                    Monitor.Pulse(Program.filelock);
                }
            }
        }
        
        public void SaveBias(Vector<double> Bias, int layer)
        {
            string filename = $"Data\\Layer {layer}\\Bias.txt";

            string[] bias = new string[Bias.Count];

            for(int i = 0;i < Bias.Count; i++)
            {
                bias[i] = Bias[i].ToString();
            }

            lock (Program.filelock)
            {
                try
                {
                    while (!Program.isfree)
                    {
                        Monitor.Wait(Program.filelock);
                    }
                    Program.isfree = false;

                    File.WriteAllText(filename, string.Join(",", bias));

                }
                finally
                {
                    Program.isfree = true;
                    Monitor.Pulse(Program.filelock);
                }
            }
        }

        /*        public void SaveLayorVectors(Vector<double> LayerVector,int Pass ,int layer)
                {
                    Console.WriteLine("making child");

                    string fileName = "Database.h5";

                    ulong numRows = Convert.ToUInt64(LayerVector.Count());
                    Console.WriteLine(numRows);

                    // Open the HDF5 file
                    var fileId = H5F.open(fileName, H5F.ACC_RDWR);



                    // Create a dataspace for the child dataset
                    uint childNumCols = 1;
                    ulong[] childDims = { numRows, childNumCols };
                    var childDataspaceId = H5S.create_simple(2, childDims, null);

                    // Create a child dataset within the file
                    var childDatasetId = H5D.create(fileId, $"child_dataset_{Pass}_layer_{layer}", H5T.NATIVE_DOUBLE, childDataspaceId);

                    // Generate sample data for the child dataset
                    double[] columnData = LayerVector.ToArray();


                    // Write data to the child dataset
                    GCHandle columnHandle = GCHandle.Alloc(columnData, GCHandleType.Pinned);
                    H5D.write(childDatasetId, H5T.NATIVE_DOUBLE, H5S.ALL, H5S.ALL, H5P.DEFAULT, columnHandle.AddrOfPinnedObject());
                    columnHandle.Free();

                    // Close resources for the child dataset
                    H5D.close(childDatasetId);
                    H5S.close(childDataspaceId);


                    // Close the HDF5 file
                    H5F.close(fileId);

                    Console.WriteLine("finished making child");
                }

                public double[] ReadSpecificChild(int targetLayer, int targetPass)
                {
                    string fileName = "Database.h5";

                    // Initialize the HDF5 library
                    H5.open();

                    // Open the HDF5 file
                    var fileId = H5F.open(fileName, H5F.ACC_RDONLY);

                    // Open the specified child dataset
                    var childDatasetId = H5D.open(fileId, $"child_dataset_{targetPass}_layer_{targetLayer}");

                    // Get the dataspace of the child dataset
                    var childDataspaceId = H5D.get_space(childDatasetId);

                    // Get the dimensions of the child dataset
                    ulong[] childDims = new ulong[2];
                    H5S.get_simple_extent_dims(childDataspaceId, childDims, null);

                    // Read data from the child dataset
                    double[] childData = new double[childDims[0] * childDims[1]];
                    GCHandle childHandle = GCHandle.Alloc(childData, GCHandleType.Pinned);
                    H5D.read(childDatasetId, H5T.NATIVE_DOUBLE, H5S.ALL, H5S.ALL, H5P.DEFAULT, childHandle.AddrOfPinnedObject());
                    childHandle.Free();

                    // Close resources for the specified child dataset
                    H5D.close(childDatasetId);
                    H5S.close(childDataspaceId);

                    // Close the HDF5 file
                    H5F.close(fileId);

                    // Close the HDF5 library
                    H5.close();

                    return childData;

                }
        */

        public int GetDimentions(int layer)
        {

            string filePath = "Dimentions.txt";
            string[] Data = File.ReadAllLines(filePath);


            string pattern = $"layer{layer}Dimention\\s*=\\s*(\\d+)";
            Regex regex = new Regex(pattern);


            Match match = regex.Match(Data[layer]);

            if (match.Success)
            {
                string dimensionValue = match.Groups[1].Value;
                if (int.TryParse(dimensionValue, out int extractedDimension))
                {
                    return extractedDimension;
                }
                else
                {
                    Console.WriteLine("Failed to convert the extracted string to an integer.");
                    return 0;
                }
            }
            else
            {
                Console.WriteLine("No match found in the string");
                return 0;
            }
            
        }

        void SetWeightDimentions(int layer)
        {
            ManageData getData = new ManageData();

            var dimention1 = getData.GetDimentions(layer);
            var dimention2 = getData.GetDimentions(layer + 1);


            weights = Matrix<double>.Build.Dense(dimention2, dimention1);

        }
        void SetBiasDimentions(int layer)
        {
            ManageData getData = new ManageData();

            var dimention2 = getData.GetDimentions(layer + 1);

            BiasVector = Vector<double>.Build.Dense(dimention2);

        }

       List<Matrix<double>> getKernel()
       {
           string Directory = $"";
       }



    }
}
