using MathNet.Numerics.LinearAlgebra;
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using Apache.Arrow.Types;
using System.Management.Instrumentation;
using System.Collections.Generic;

namespace UpdatedProject
{
    internal class NetInIt
    {

        public static ImageHandle imageHandle = new ImageHandle();

        public Matrix<double> weights;



        public Vector<double> BiasVector;


        public NetInIt(int Pass, int layer)
        {

            FileGen(Pass, layer);

            LayerGen(Pass, layer - 1);

        }



        public NetInIt()
        {

        }
        //heheheha

        void ParguetGen()
        {
            // Generate sample data
            var data = GenerateSampleData();

            // Create schema with 1000 columns
            var schema = CreateSchema(1000);

            // Create Arrow table with 3 records
            var table = CreateArrowTable(schema, data);

            // Write Arrow table to Parquet file
            WriteToParquet(table, "output.parquet");

            Console.WriteLine("Parquet file created successfully.");
        }
        static List<Vector<double>> GenerateSampleData()
        {
            // Generate sample data (replace with your own logic)
            var data = new List<Vector<double>>();
            var rng = new Random();

            for (int i = 0; i < 3; i++)
            {
                var vector = Vector<double>.Build.Dense(1000, j => rng.NextDouble());
                data.Add(vector);
            }

            return data;
        }

        static Schema CreateSchema(int numColumns)
        {
            var fields = new List<Field>();

            for (int i = 0; i < numColumns; i++)
            {
                var fieldName = $"layer_{i + 1}";
                var metadata = new Dictionary<string, string>
                {
                    { "description", $"Description for {fieldName}" },
                    { "customMetadata", "Any custom metadata here" }
                 // Add more metadata as needed
                };

                var field = new Field(fieldName, new ListType(FloatType.Default), false, metadata);
                fields.Add(field);
            }

            return new Schema(fields);
        }

        static Table CreateArrowTable(Schema schema, List<Vector<double>> data)
        {
            // Create Arrow table with the specified schema and data
            var recordBatches = new List<RecordBatch>();

            for (int i = 0; i < data.Count; i++)
            {
                var vectors = data[i];
                var recordBatchBuilder = new RecordBatchBuilder(schema);

                for (int j = 0; j < schema.Fields.Count; j++)
                {
                    recordBatchBuilder.GetFieldBuilder<ListBuilder<double>>(j).Append(vectors[j]);
                }

                var recordBatch = recordBatchBuilder.Build();
                recordBatches.Add(recordBatch);
            }

            return new Table(schema, recordBatches);
        }

        static void WriteToParquet(Table table, string outputPath)
        {
            // Write Arrow table to Parquet file
            using (var fileWriter = new ParquetFileWriter(outputPath, table.Schema))
            {
                fileWriter.WriteTable(table);
            }
        }



        static public int GetFileDimentions(int Pass)
        {
            string filename = $"image_{Pass}_";
            string DimFile = "Dimentions.txt";


            Vector<double> vals = imageHandle.NormRGB(filename, Pass);



            string existingText;
            using (StreamReader reader = new StreamReader(DimFile))
            {
                existingText = reader.ReadToEnd();
            }

            // Write the new text and then the existing content back to the file
            using (StreamWriter writer = new StreamWriter(DimFile))
            {
                writer.Write($"layer0Dimention = {vals.Count}");
                writer.Write(existingText);
            }

            return vals.Count;
        }


        public void FileGen(int Pass, int layer)
        {
            for (int i = 0; i <= Pass; i++)
            {
                string filepath = $"Data\\Pass {i}\\Output";

                if (!File.Exists(filepath))
                {
                    try
                    {
                        Directory.CreateDirectory(filepath);
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine($"Error occured: {e}");
                    }
                }
            }
            for (int i = 0; i < layer; i++)
            {
                string filepath = $"Data\\Layer {i}";

                if (!File.Exists(filepath))
                {
                    try
                    {
                        Directory.CreateDirectory(filepath);
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine($"Error occured: {e}");
                    }
                }
            }


        }
        void LayerGen(int Pass, int layer)
        {
            for (int i = 0; i <= layer; i++)
            {
                WeightGen(i);
                BiasGen(i);
            }


        }



        public void WeightGen(int layer)
        {
            SetWeightDimentions(layer);

            GaussianRandomGenerator generator = new GaussianRandomGenerator();

            double mean = 0.0;
            double stdDev = 1.0;
            string Filename = $"Data\\Layer {layer}\\Weights.txt";
            int k = 0;



            if (File.Exists(Filename))
            {
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

            }
            else
            {
                string[] vals = new string[weights.ColumnCount * weights.RowCount];
                for (int i = 0; i < weights.ColumnCount * weights.RowCount; i++)
                {
                    vals[i] = Convert.ToString(generator.Generate(mean, stdDev));
                }

                try
                {
                    File.WriteAllText(Filename, string.Join(",", vals));
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"An error occurred: {ex.Message}");

                }




                System.Array.Clear(vals, 0, vals.Length);

                string content = File.ReadAllText(Filename);
                vals = content.Split(',');
                for (int i = 0; i < weights.ColumnCount; i++)
                {
                    for (int j = 0; j < weights.RowCount; j++)
                    {
                        weights[j, i] = Convert.ToDouble(vals[k]);
                        k++;
                    }
                }


            }
        }


        public void BiasGen(int layer)
        {
            SetBiasDimentions(layer);

            string filename = $"Data\\layer {layer}\\Bias.txt";
            int k = 0;

            if (File.Exists(filename))
            {

                string content = File.ReadAllText(filename);
                string[] vals = content.Split(',');
                for (int i = 0; i < BiasVector.Count; i++)
                {
                    BiasVector[i] = Convert.ToDouble(vals[k]);
                    k++;
                }
            }
            else
            {
                string[] vals = new string[BiasVector.Count];
                for (int i = 0; i < BiasVector.Count; i++)
                {
                    vals[i] = "0";
                }

                try
                {
                    File.WriteAllText(filename, string.Join(",", vals));
                }
                catch (IOException ex)
                {
                    Console.WriteLine($"An error occurred: {ex.Message}");

                }


                System.Array.Clear(vals, 0, vals.Length);

                string content = File.ReadAllText(filename);
                vals = content.Split(',');
                for (int i = 0; i < BiasVector.Count; i++)
                {
                    BiasVector[i] = Convert.ToDouble(vals[k]);
                    k++;
                }


            }
        }


        void SetWeightDimentions(int layer)
        {
            GetData getData = new GetData();

            var dimention1 = getData.GetDimentions(layer);
            var dimention2 = getData.GetDimentions(layer + 1);


            weights = Matrix<double>.Build.Dense(dimention2, dimention1);

        }
        void SetBiasDimentions(int layer)
        {
            GetData getData = new GetData();

            var dimention2 = getData.GetDimentions(layer + 1);

            BiasVector = Vector<double>.Build.Dense(dimention2);

        }


    }
    public class GaussianRandomGenerator
    {
        private Random random;

        public GaussianRandomGenerator()
        {
            random = new Random();
        }

        public GaussianRandomGenerator(int seed)
        {
            random = new Random(seed);
        }

        public double Generate(double mean, double stdDev)
        {
            // Generate two random numbers between 0 and 1
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();

            // Apply the Box-Muller transform to generate a standard Gaussian random number
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

            // Scale and shift the standard Gaussian random number to have the desired mean and stdDev
            double result = mean + z * stdDev;

            return result;
        }

    }

}
