using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;

namespace UpdatedProject
{
    internal class Program
    {
        const int Xweight = 784;
        const int Yweight = 16;




        static void Main(string[] args)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            Datainit();

            Console.WriteLine("enter max image count");
            //int imageMaxIndex = Convert.ToInt32(Console.ReadLine());
            int Passes = 59999;

            pass(Passes);
            


            Console.WriteLine(sw.Elapsed.ToString());
        }



        static void pass(int Passes)
        {
            int imageMaxIndex = 2;

            Console.WriteLine("Processing...");
            NetInIt networkGen = new NetInIt(Passes, imageMaxIndex);

            Parallel.For(0, Passes + 1, i =>
            {
                if (!File.Exists($"Data\\Pass {i}\\Output\\LayerVector.txt"))
                {
                    ForwardPass forwardPass = new ForwardPass(i);
                    forwardPass.Forwards(forwardPass.LayerVector, i, 0, imageMaxIndex);
                }
            });

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
