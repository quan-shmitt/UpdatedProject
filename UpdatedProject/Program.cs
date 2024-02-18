﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace UpdatedProject
{
    internal class Program
    {
        const int Xweight = 784;
        const int Yweight = 16;

        static int LayerCount = 2;



        static public object filelock = new object();
        static public bool isfree = true;

        static public double cost = 0;


        static void Main(string[] args)
        {
            ManageData manageData = new ManageData();

            Stopwatch sw = new Stopwatch();
            sw.Start();

            Datainit();

            Console.WriteLine("Test or Train: \n");
            string ProcessDirector = Console.ReadLine();
            if(ProcessDirector.ToUpper() == "TEST")
            {
                PredictInput predictInput = new PredictInput();
                predictInput.FindNumInPicture(LayerCount);
            }
            else
            {
                Console.WriteLine("enter max image count");

                int Passes = 100;
                int epochs = 4;

                pass(Passes, epochs);
            }

            Console.WriteLine(sw.Elapsed.ToString());
        }



        static void pass(int Passes, int epoch)
        {
            ManageData manageData = new ManageData();


            ImageHandle image = new ImageHandle();  

            Console.WriteLine("Processing...");
            NetInIt networkGen = new NetInIt(Passes, LayerCount, Passes);

            for (int j = 0; j < epoch; j++)
            {
                Parallel.For(0, Passes + 1, i =>
                {
                    MLP forwardPass = new MLP(i, LayerCount);
                    if (!File.Exists($"Data\\Pass {i}\\Output\\LayerVector.txt"))
                    {
                        forwardPass.Forwards(forwardPass.LayerVector, 0, LayerCount);
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
