﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace UpdatedProject
{
    internal class Program
    {

        static int LayerCount = 2;
        static int CNNCount = 2;



        static public object filelock = new object();
        static public bool isfree = true;

        static public double cost = 0;


        static void Main(string[] args)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            Datainit();

            Console.WriteLine("Test or Train: \n");
            string ProcessDirector = Console.ReadLine();
            if (ProcessDirector.ToUpper() == "TEST")
            {
                PredictInput predictInput = new PredictInput();
                predictInput.FindNumInPicture(LayerCount);
            }
            else
            {
                Console.WriteLine("enter max image count");

                int Passes = 75;
                int epochs = 4;

                Pass(Passes, epochs);
            }

            Console.WriteLine(sw.Elapsed.ToString());
        }



        static void Pass(int Passes, int epoch)
        {
            ManageData manageData = new ManageData();


            ImageHandle image = new ImageHandle();

            Console.WriteLine("Processing...");
            NetInIt networkGen = new NetInIt(Passes, LayerCount, CNNCount);

            for (int j = 0; j < epoch; j++)
            {
                Parallel.For(0, Passes + 1, i =>
                {
                    MLP forwardPass = new MLP(i, LayerCount);
                    CNNLayers CNN = new CNNLayers();
                    if (!File.Exists($"Data\\Pass {i}\\Output\\LayerVector.txt"))
                    {
                        CNN.Forwards(i, LayerCount, 200);
                        forwardPass.Forwards(CNN.MatrixToVector(CNN.result), 0, LayerCount);
                    }
                    Backpropagation backpropagation = new Backpropagation(LayerCount);

                    List<Vector<double>> Input = forwardPass.Cache;

                    backpropagation.BackProp(Input, image.Label(Convert.ToInt32(i), 10), 0.05, LayerCount);
                });
                Console.WriteLine(cost);
                cost = 0;
            }


            Console.WriteLine("Finished");
        }



        static void Datainit()
        {
            Console.WriteLine("started");
            string path = "MNIST\\MNISTDataSet.exe";

            if (!File.Exists("MNIST\\images"))
            {
                Console.WriteLine("started processing");
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = path,
                    UseShellExecute = true
                };

                try
                {
                    Process process = Process.Start(startInfo);
                    process.WaitForExit();
                    Console.WriteLine("finished");
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                }
            }
        }

    }
}
